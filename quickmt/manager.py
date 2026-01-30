import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import OrderedDict
from functools import lru_cache

from fastapi import HTTPException
from huggingface_hub import HfApi, snapshot_download
from cachetools import TTLCache, cached, LRUCache

from quickmt.translator import Translator
from quickmt.settings import settings

logger = logging.getLogger(__name__)


class BatchTranslator:
    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str = "cpu",
        compute_type: str = "default",
        inter_threads: int = 1,
        intra_threads: int = 0,
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.inter_threads = inter_threads
        self.intra_threads = intra_threads
        self.translator: Optional[Translator] = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        # LRU cache for translations
        self.translation_cache: LRUCache = LRUCache(
            maxsize=settings.translation_cache_size
        )

    async def start_worker(self):
        if self.worker_task:
            return

        # Load model in main process (or worker thread if needed)
        # For now, Translator handles its own loading
        self.translator = Translator(
            Path(self.model_path),
            device=self.device,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            intra_threads=self.intra_threads,
        )
        self.worker_task = asyncio.create_task(self._worker())
        logger.info(f"Started translation worker for model: {self.model_id}")

    async def stop_worker(self):
        if not self.worker_task:
            return

        # Send sentinel to stop worker
        await self.queue.put(None)
        await self.worker_task
        self.worker_task = None
        if self.translator:
            self.translator.unload()
            self.translator = None
        logger.info(f"Stopped translation worker for model: {self.model_id}")

    async def _worker(self):
        while True:
            item = await self.queue.get()
            if item is None:
                self.queue.task_done()
                break

            src, src_lang, tgt_lang, kwargs, future = item
            try:
                # 1. Collect batch
                batch_texts = [src]
                futures = [future]

                # Try to grab more items up to MAX_BATCH_SIZE or timeout
                start_time = time.time()
                while len(batch_texts) < settings.max_batch_size:
                    wait_time = (settings.batch_timeout_ms / 1000.0) - (
                        time.time() - start_time
                    )
                    if wait_time <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(
                            self.queue.get(), timeout=wait_time
                        )
                        if next_item is None:
                            # Re-add sentinel to handle later
                            await self.queue.put(None)
                            break
                        n_src, n_sl, n_tl, n_kw, n_fut = next_item

                        # Only batch if parameters match exactly
                        if n_sl == src_lang and n_tl == tgt_lang and n_kw == kwargs:
                            batch_texts.append(n_src)
                            futures.append(n_fut)
                        else:
                            # Re-queue item for a later batch/worker cycle
                            await self.queue.put(next_item)
                            break
                    except asyncio.TimeoutError:
                        break

                # 2. Process batch
                # Run in executor to avoid blocking the asyncio loop during inference
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self.translator(
                        batch_texts, src_lang=src_lang, tgt_lang=tgt_lang, **kwargs
                    ),
                )

                # result can be string or list
                if isinstance(results, str):
                    results = [results]

                # 3. Resolve futures
                for res, fut in zip(results, futures):
                    if not fut.done():
                        fut.set_result(res)

                # Mark done for all processed items
                for _ in range(len(batch_texts)):
                    self.queue.task_done()

            except Exception as e:
                logger.error(f"Error in translation worker for {self.model_id}: {e}")
                if not future.done():
                    future.set_exception(e)
                # TODO: handle others if batched

    async def translate(
        self, src: str, src_lang: str = None, tgt_lang: str = None, **kwargs
    ) -> str:
        if not self.worker_task:
            await self.start_worker()

        # Create cache key from input parameters
        # Convert kwargs to a sorted tuple for hashability
        kwargs_tuple = tuple(sorted(kwargs.items()))
        cache_key = (src, src_lang, tgt_lang, kwargs_tuple)

        # Check cache first
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        # Cache miss - perform translation
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((src, src_lang, tgt_lang, kwargs, future))
        result = await future

        # Store in cache
        self.translation_cache[cache_key] = result
        return result


class ModelManager:
    def __init__(
        self,
        max_loaded: int,
        device: str,
        compute_type: str = "default",
        inter_threads: int = 1,
        intra_threads: int = 0,
    ):
        self.max_loaded = max_loaded
        self.device = device
        self.compute_type = compute_type
        self.inter_threads = inter_threads
        self.intra_threads = intra_threads
        # cache key: src-tgt string
        self.models: OrderedDict[str, BatchTranslator] = OrderedDict()
        self.pending_loads: Dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()
        self.hf_collection_models: List[Dict] = []
        self.api = HfApi()

    @cached(cache=TTLCache(maxsize=1, ttl=3600))
    async def fetch_hf_models(self):
        """Fetch available models from the quickmt collection on Hugging Face."""
        try:
            loop = asyncio.get_running_loop()
            collection = await loop.run_in_executor(
                None, lambda: self.api.get_collection("quickmt/quickmt-models")
            )

            hf_models = []
            for item in collection.items:
                if item.item_type == "model":
                    model_id = item.item_id
                    # Expecting format: quickmt/quickmt-en-fr
                    parts = model_id.split("/")[-1].replace("quickmt-", "").split("-")
                    if len(parts) == 2:
                        src, tgt = parts
                        hf_models.append(
                            {"model_id": model_id, "src_lang": src, "tgt_lang": tgt}
                        )
            self.hf_collection_models = hf_models
            logger.info(
                f"Discovered {len(hf_models)} models from Hugging Face collection"
            )
        except Exception as e:
            logger.error(f"Failed to fetch models from Hugging Face: {e}")

    async def get_model(self, src_lang: str, tgt_lang: str) -> BatchTranslator:
        model_name = f"{src_lang}-{tgt_lang}"

        async with self.lock:
            # 1. Check if loaded
            if model_name in self.models:
                self.models.move_to_end(model_name)
                return self.models[model_name]

            # 2. Check if currently loading
            if model_name in self.pending_loads:
                event = self.pending_loads[model_name]
            else:
                # NEW: Pre-check existence before starting task to ensure clean 404
                hf_model = next(
                    (
                        m
                        for m in self.hf_collection_models
                        if m["src_lang"] == src_lang and m["tgt_lang"] == tgt_lang
                    ),
                    None,
                )
                if not hf_model:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model for {src_lang}->{tgt_lang} not found in Hugging Face collection",
                    )

                event = asyncio.Event()
                self.pending_loads[model_name] = event
                # This task will do the actual loading
                asyncio.create_task(self._load_model_task(src_lang, tgt_lang, event))

        # 3. Wait for load
        await event.wait()

        # 4. Return from cache
        async with self.lock:
            return self.models[model_name]

    async def _load_model_task(
        self, src_lang: str, tgt_lang: str, new_event: asyncio.Event
    ):
        model_name = f"{src_lang}-{tgt_lang}"
        try:
            try:
                # Find matching model from HF collection (already checked in get_model)
                hf_model = next(
                    m
                    for m in self.hf_collection_models
                    if m["src_lang"] == src_lang and m["tgt_lang"] == tgt_lang
                )

                logger.info(f"Accessing Hugging Face model: {hf_model['model_id']}")
                loop = asyncio.get_running_loop()
                # snapshot_download returns the local path in the HF cache.
                # Try local only first to speed up loading
                try:
                    cached_path = await loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=hf_model["model_id"],
                            ignore_patterns=["eole-model/*", "eole_model/*"],
                            local_files_only=True,
                        ),
                    )
                except Exception:
                    # Fallback to checking online
                    logger.info(
                        f"Model {hf_model['model_id']} not fully cached, checking online..."
                    )
                    cached_path = await loop.run_in_executor(
                        None,
                        lambda: snapshot_download(
                            repo_id=hf_model["model_id"],
                            ignore_patterns=["eole-model/*", "eole_model/*"],
                        ),
                    )
                model_path = Path(cached_path)

                # Prepare for eviction
                evicted_model = None
                async with self.lock:
                    if len(self.models) >= self.max_loaded:
                        oldest_name, evicted_model = self.models.popitem(last=False)
                        logger.info(f"Evicting model: {oldest_name}")

                if evicted_model:
                    await evicted_model.stop_worker()

                # Load new model (SLOW, outside lock)
                logger.info(
                    f"Loading model: {hf_model['model_id']} (device: {self.device}, compute: {self.compute_type})"
                )
                new_model = BatchTranslator(
                    model_id=hf_model["model_id"],
                    model_path=str(model_path),
                    device=self.device,
                    compute_type=self.compute_type,
                    inter_threads=self.inter_threads,
                    intra_threads=self.intra_threads,
                )
                await new_model.start_worker()

                # Add to cache
                async with self.lock:
                    self.models[model_name] = new_model

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                # We still need to set the event to unblock waiters,
                # but we should probably handle errors better in get_model
                raise e
        finally:
            async with self.lock:
                if model_name in self.pending_loads:
                    del self.pending_loads[model_name]
                    new_event.set()

    def list_available_models(self) -> List[Dict]:
        """List all models discovered from Hugging Face."""
        available = []
        for m in self.hf_collection_models:
            lang_pair = f"{m['src_lang']}-{m['tgt_lang']}"
            available.append(
                {
                    "model_id": m["model_id"],
                    "src_lang": m["src_lang"],
                    "tgt_lang": m["tgt_lang"],
                    "loaded": lang_pair in self.models,
                }
            )
        return available

    @lru_cache(maxsize=1)
    def get_language_pairs(self) -> Dict[str, List[str]]:
        """Return a dictionary of source languages to list of supported target languages."""
        pairs: Dict[str, set] = {}
        for m in self.hf_collection_models:
            src = m["src_lang"]
            tgt = m["tgt_lang"]
            if src not in pairs:
                pairs[src] = set()
            pairs[src].add(tgt)

        # Convert sets to sorted lists
        return {src: sorted(list(tgts)) for src, tgts in sorted(pairs.items())}

    async def shutdown(self):
        for name, model in self.models.items():
            await model.stop_worker()
        self.models.clear()
