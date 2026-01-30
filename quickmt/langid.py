from typing import List, Tuple, Union, Optional
from pathlib import Path
import os
import urllib.request
import fasttext

# Suppress fasttext's warning about being loaded in a way that doesn't 
# allow querying its version (common in some environments)
fasttext.FastText.eprint = lambda x: None


class LanguageIdentification:
    """Detect language using a FastText langid model.
    
    This class provides a wrapper around the FastText library for efficient 
    language identification, supporting both single-string and batch processing.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize the LanguageIdentification model.

        Args:
            model_path: Path to the pre-trained FastText model file.
                 If None, defaults to 'models/lid.176.bin' and downloads if missing.
        """
        if model_path is None:
            cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
            model_dir = cache_dir / "fasttext_language_id"
            model_path = model_dir / "lid.176.bin"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            print(f"Downloading FastText model from {url} to {model_path}...")
            urllib.request.urlretrieve(url, str(model_path))
            print("Download complete.")
            
        self.ft = fasttext.load_model(str(model_path))

    def predict(
        self, 
        text: Union[str, List[str]], 
        k: int = 1,
        threshold: float = 0.0
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """Predict the language(s) for the given text or list of texts.

        Args:
            text: A single string or a list of strings to identify.
            k: Number of most likely languages to return. Defaults to 1.
            threshold: Minimum score for a language to be included in the results.
                Defaults to 0.0 (return all k results regardless of score).

        Returns:
            If input is a string: A list of (lang, score) tuples.
            If input is a list of strings: A list of lists of (lang, score) tuples, 
                maintaining the input order.
        """
        is_single = isinstance(text, str)
        items = [text] if is_single else text
        
        # Sanitize inputs: FastText errors on newlines
        items = [t.replace("\n", " ") for t in items]
        
        # FastText predict handles lists natively and is faster than looping
        ft_output = self.ft.predict(items, k=k, threshold=threshold)
        
        # FastText returns ([['__label__en', ...], ...], [[0.9, ...], ...])
        labels, scores = ft_output
        
        results = []
        for item_labels, item_scores in zip(labels, scores):
            item_results = [
                (label.replace("__label__", ""), float(score))
                for label, score in zip(item_labels, item_scores)
            ]
            results.append(item_results)
            
        return results[0] if is_single else results

    def predict_best(
        self, 
        text: Union[str, List[str]], 
        threshold: float = 0.0
    ) -> Union[Optional[str], List[Optional[str]]]:
        """Predict the most likely language for the given text or list of texts.

        This is a convenience wrapper around `predict` that returns only the
        top-scoring language label (or None if no language exceeds the threshold).

        Args:
            text: A single string or a list of strings to identify.
            threshold: Minimum score for a language to be selected.

        Returns:
            If input is a string: The language code (e.g., 'en') or None.
            If input is a list: A list of language codes or None.
        """
        results = self.predict(text, k=1, threshold=threshold)
        
        if isinstance(text, str):
            # results is List[Tuple[str, float]]
            return results[0][0] if results else None
        else:
            # results is List[List[Tuple[str, float]]]
            return [r[0][0] if r else None for r in results]


def ensure_model_exists(model_path: Optional[Union[str, Path]] = None):
    """Ensure the FastText model exists on disk, downloading if necessary.
    This should be called from the main process before starting worker pools.
    """
    if model_path is None:
        cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        model_dir = cache_dir / "fasttext_language_id"
        model_path = model_dir / "lid.176.bin"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        print(f"Downloading FastText model from {url} to {model_path}...")
        urllib.request.urlretrieve(url, str(model_path))
        print("Download complete.")


# Global detector instance for process pool workers
_detector: Optional[LanguageIdentification] = None


def init_worker(model_path: Optional[Union[str, Path]] = None):
    """Initialize the global detector instance for a worker process."""
    global _detector
    # We assume ensure_model_exists was already called in the main process
    _detector = LanguageIdentification(model_path)


def predict_worker(
    text: Union[str, List[str]], 
    k: int = 1, 
    threshold: float = 0.0
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Prediction function to be run in a worker process."""
    if _detector is None:
        # Fallback if init_worker failed or wasn't called
        init_worker()
    return _detector.predict(text, k=k, threshold=threshold)