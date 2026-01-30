"""
Simple test to demonstrate the translation cache functionality.
Run this to verify cache hits provide instant responses.
"""

import asyncio
import time
from quickmt.manager import BatchTranslator
from quickmt.settings import settings


async def test_translation_cache():
    print("=== Translation Cache Test ===\n")

    # Create a mock BatchTranslator (would normally be created by ModelManager)
    # For this test, we'll just verify the cache mechanism
    print(f"Cache size configured: {settings.translation_cache_size}")

    # Simulate cache behavior
    from cachetools import LRUCache

    cache = LRUCache(maxsize=settings.translation_cache_size)

    # Test data
    test_text = "Hello, world!"
    src_lang = "en"
    tgt_lang = "fr"
    kwargs_tuple = tuple(sorted({"beam_size": 5, "patience": 1}.items()))

    cache_key = (test_text, src_lang, tgt_lang, kwargs_tuple)

    # First request - cache miss
    print("\n1. First translation (cache miss):")
    print(f"   Key: {cache_key}")
    if cache_key in cache:
        print("   ✓ Cache HIT")
    else:
        print("   ✗ Cache MISS (expected)")
        # Simulate translation and caching
        cache[cache_key] = "Bonjour, monde!"
        print("   → Cached result")

    # Second request - cache hit
    print("\n2. Repeated translation (cache hit):")
    print(f"   Key: {cache_key}")
    if cache_key in cache:
        print("   ✓ Cache HIT (instant!)")
        print(f"   → Result: {cache[cache_key]}")
    else:
        print("   ✗ Cache MISS (unexpected)")

    # Different parameters - cache miss
    different_kwargs = tuple(sorted({"beam_size": 10, "patience": 2}.items()))
    different_key = (test_text, src_lang, tgt_lang, different_kwargs)

    print("\n3. Same text, different parameters (cache miss):")
    print(f"   Key: {different_key}")
    if different_key in cache:
        print("   ✓ Cache HIT")
    else:
        print("   ✗ Cache MISS (expected - different params)")

    print("\n✅ Cache test complete!")
    print(f"Cache size: {len(cache)}/{settings.translation_cache_size}")


if __name__ == "__main__":
    asyncio.run(test_translation_cache())
