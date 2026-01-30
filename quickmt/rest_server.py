import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Union, Dict
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from quickmt.langid import init_worker, predict_worker, ensure_model_exists
from quickmt.manager import ModelManager
from quickmt.settings import settings

logger = logging.getLogger("uvicorn.error")


class TranslationRequest(BaseModel):
    src: Union[str, List[str]]
    src_lang: Optional[Union[str, List[str]]] = None
    tgt_lang: str = "en"
    beam_size: int = 5
    patience: int = 1
    length_penalty: float = 1.0
    coverage_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_decoding_length: int = 256

    @model_validator(mode="after")
    def validate_patience(self):
        if self.patience > self.beam_size:
            raise ValueError("patience cannot be greater than beam_size")
        return self


class TranslationResponse(BaseModel):
    translation: Union[str, List[str]]
    src_lang: Union[str, List[str]]
    src_lang_score: Union[float, List[float]]
    tgt_lang: str
    processing_time: float
    model_used: Union[str, List[str]]


class DetectionRequest(BaseModel):
    src: Union[str, List[str]]
    k: int = 1
    threshold: float = 0.0


class DetectionResult(BaseModel):
    lang: str
    score: float


class DetectionResponse(BaseModel):
    results: Union[List[DetectionResult], List[List[DetectionResult]]]
    processing_time: float


class BatchItem:
    def __init__(
        self,
        src: List[str],
        src_lang: str,
        tgt_lang: str,
        beam_size: int,
        max_decoding_length: int,
        future: asyncio.Future,
    ):
        self.src = src
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.beam_size = beam_size
        self.max_decoding_length = max_decoding_length
        self.future = future


# Global instances initialized in lifespan
model_manager: Optional[ModelManager] = None
langid_executor: Optional[ProcessPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, langid_executor

    model_manager = ModelManager(
        max_loaded=settings.max_loaded_models,
        device=settings.device,
        compute_type=settings.compute_type,
        inter_threads=settings.inter_threads,
        intra_threads=settings.intra_threads,
    )

    # 1. Fetch available models from Hugging Face
    await model_manager.fetch_hf_models()

    # 2. Ensure langid model is downloaded in main process before starting workers
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ensure_model_exists, settings.langid_model_path)

    # Initialize langid process pool
    langid_executor = ProcessPoolExecutor(
        max_workers=settings.langid_workers,
        initializer=init_worker,
        initargs=(settings.langid_model_path,),
    )

    yield

    if langid_executor:
        langid_executor.shutdown()
    await model_manager.shutdown()


app = FastAPI(
    title="quickmt Multi-Model API",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
api_router = APIRouter(prefix="/api")


@api_router.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")

    start_time = time.time()
    src_list = [request.src] if isinstance(request.src, str) else request.src
    if not src_list:
        return TranslationResponse(
            translation="" if isinstance(request.src, str) else [],
            src_lang="" if isinstance(request.src, str) else [],
            src_lang_score=0.0 if isinstance(request.src, str) else [],
            tgt_lang=request.tgt_lang,
            processing_time=time.time() - start_time,
            model_used="none",
        )

    try:
        loop = asyncio.get_running_loop()

        # 1. Determine source languages and confidence scores
        if request.src_lang:
            if isinstance(request.src_lang, list):
                if not isinstance(src_list, list) or len(request.src_lang) != len(
                    src_list
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="src_lang list length must match src list length",
                    )
                src_langs = request.src_lang
                src_lang_scores = [1.0] * len(src_list)
            else:
                src_langs = [request.src_lang] * len(src_list)
                src_lang_scores = [1.0] * len(src_list)
        else:
            if not langid_executor:
                raise HTTPException(
                    status_code=503, detail="Language identification not initialized"
                )
            # Batch detect languages
            raw_langid_results = await loop.run_in_executor(
                langid_executor,
                predict_worker,
                src_list,
                1,  # k=1 (best guess)
                0.0,  # threshold
            )
            # results are List[List[Tuple[str, float]]], extract labels and scores
            src_langs = [r[0][0] if r else "unknown" for r in raw_langid_results]
            src_lang_scores = [float(r[0][1]) if r else 0.0 for r in raw_langid_results]

        # 2. Group indices by source language
        # groups: { "fr": [0, 2, ...], "es": [1, ...] }
        groups: Dict[str, List[int]] = {}
        for idx, lang in enumerate(src_langs):
            if lang not in groups:
                groups[lang] = []
            groups[lang].append(idx)

        # 3. Process each group
        final_translations = [""] * len(src_list)
        final_models = [""] * len(src_list)
        tasks = []

        # We need a way to track which lang pairs were actually used for the 'model_used' string
        used_pairs = set()

        for lang, indices in groups.items():
            group_src = [src_list[i] for i in indices]

            # Optimization: If src == tgt, skip translation
            if lang == request.tgt_lang:
                for src_idx, idx in enumerate(indices):
                    final_translations[idx] = group_src[src_idx]
                    final_models[idx] = "identity"
                continue

            # Load model and translate for this group
            async def process_group_task(l=lang, i_list=indices, g_src=group_src):
                try:
                    translator = await model_manager.get_model(l, request.tgt_lang)
                    used_pairs.add(translator.model_id)
                    # Call translate for each sentence; BatchTranslator will handle opportunistic batching
                    translation_tasks = [
                        translator.translate(
                            s,
                            src_lang=l,
                            tgt_lang=request.tgt_lang,
                            beam_size=request.beam_size,
                            patience=request.patience,
                            length_penalty=request.length_penalty,
                            coverage_penalty=request.coverage_penalty,
                            repetition_penalty=request.repetition_penalty,
                            max_decoding_length=request.max_decoding_length,
                        )
                        for s in g_src
                    ]
                    results = await asyncio.gather(*translation_tasks)
                    for result_idx, original_idx in enumerate(i_list):
                        final_translations[original_idx] = results[result_idx]
                        final_models[original_idx] = translator.model_id
                except HTTPException as e:
                    # If a specific model is missing, we could either fail the whole batch
                    # or keep original text. Here we fail for consistency with previous behavior.
                    raise e
                except Exception as e:
                    logger.error(f"Error translating {l} to {request.tgt_lang}: {e}")
                    raise e

            tasks.append(process_group_task())

        if tasks:
            await asyncio.gather(*tasks)

        # 4. Prepare response
        if isinstance(request.src, str):
            result = final_translations[0]
            src_lang_res = src_langs[0]
            src_lang_score_res = src_lang_scores[0]
            model_used_res = final_models[0]
        else:
            result = final_translations
            src_lang_res = src_langs
            src_lang_score_res = src_lang_scores
            model_used_res = final_models

        return TranslationResponse(
            translation=result,
            src_lang=src_lang_res,
            src_lang_score=src_lang_score_res,
            tgt_lang=request.tgt_lang,
            processing_time=time.time() - start_time,
            model_used=model_used_res,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in translate_endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/identify-language", response_model=DetectionResponse)
async def identify_language_endpoint(request: DetectionRequest):
    if not langid_executor:
        raise HTTPException(
            status_code=503, detail="Language identification not initialized"
        )

    start_time = time.time()
    try:
        loop = asyncio.get_running_loop()
        # Offload detection to process pool to avoid GIL issues
        raw_results = await loop.run_in_executor(
            langid_executor, predict_worker, request.src, request.k, request.threshold
        )

        # Convert raw tuples to Pydantic models
        if isinstance(request.src, str):
            results = [
                DetectionResult(lang=lang, score=score) for lang, score in raw_results
            ]
        else:
            results = [
                [
                    DetectionResult(lang=lang, score=score)
                    for lang, score in item_results
                ]
                for item_results in raw_results
            ]

        return DetectionResponse(
            results=results, processing_time=time.time() - start_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/models")
async def get_models():
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return {"models": model_manager.list_available_models()}


@api_router.get("/languages")
async def get_languages():
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager.get_language_pairs()


@api_router.get("/health")
async def health_check():
    loaded_models = list(model_manager.models.keys()) if model_manager else []
    return {
        "status": "ok",
        "loaded_models": loaded_models,
        "max_models": settings.max_loaded_models,
    }


app.include_router(api_router)

# Serve static files for the GUI
static_dir = os.path.join(os.path.dirname(__file__), "gui", "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


def start():
    """Entry point for the quickmt-serve CLI."""
    import uvicorn

    uvicorn.run(
        "quickmt.rest_server:app", host="0.0.0.0", port=settings.port, reload=False
    )


def start_gui():
    """Entry point for the quickmt-gui CLI."""
    import uvicorn
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://127.0.0.1:{settings.port}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "quickmt.rest_server:app", host="0.0.0.0", port=settings.port, reload=False
    )
