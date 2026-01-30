import pytest
from httpx import AsyncClient
import sacrebleu

# 10 Diverse English -> French pairs
EN_FR_PAIRS = [
    ("Hello world", "Bonjour le monde"),
    ("The cat sits on the mat.", "Le chat est assis sur le tapis."),
    ("I would like a coffee, please.", "Je voudrais un café, s'il vous plaît."),
    ("Where is the nearest train station?", "Où est la gare la plus proche ?"),
    (
        "Artificial intelligence is fascinating.",
        "L'intelligence artificielle est fascinante.",
    ),
    ("Can you help me translate this?", "Pouvez-vous m'aider à traduire ceci ?"),
    ("It is raining today.", "Il pleut aujourd'hui."),
    ("Programming is fun.", "La programmation est amusante."),
    ("I am learning French.", "J'apprends le français."),
    ("Have a nice day.", "Bonne journée."),
]

# 10 Diverse French -> English pairs
FR_EN_PAIRS = [
    ("Bonjour tout le monde", "Hello everyone"),
    ("La vie est belle", "Life is beautiful"),
    ("Je suis fatigué", "I am tired"),
    ("Quelle heure est-il ?", "What time is it?"),
    ("J'aime manger des croissants", "I like eating croissants"),
    ("Merci beaucoup", "Thank you very much"),
    ("À bientôt", "See you soon"),
    ("Le livre est sur la table", "The book is on the table"),
    ("Je ne comprends pas", "I do not understand"),
    ("C'est magnifique", "It is magnificent"),
]


async def translate_batch(client, texts, src, tgt):
    payload = {"src": texts, "src_lang": src, "tgt_lang": tgt}
    response = await client.post("/api/translate", json=payload)
    if response.status_code != 200:
        return []
    return response.json()["translation"]


@pytest.mark.asyncio
async def test_quality_en_fr(client: AsyncClient):
    """Assess translation quality for English to French."""
    # Check model availability first
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]
    if not any(
        m["src_lang"] == "en" and m["tgt_lang"] == "fr" for m in available_models
    ):
        pytest.skip("en-fr model needed for this test")

    sources = [p[0] for p in EN_FR_PAIRS]
    refs = [
        [p[1] for p in EN_FR_PAIRS]
    ]  # sacrebleu expects list of lists of references

    hyps = await translate_batch(client, sources, "en", "fr")
    assert len(hyps) == len(sources)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    chrf = sacrebleu.corpus_chrf(hyps, refs)

    print(f"\nEN->FR Quality: BLEU={bleu.score:.2f}, CHRF={chrf.score:.2f}")

    # Assert minimum quality (adjust baselines based on model capability)
    # Generic models should at least get > 10 BLEU on simple sentences
    assert bleu.score > 40.0
    assert chrf.score > 70.0


@pytest.mark.asyncio
async def test_quality_fr_en(client: AsyncClient):
    """Assess translation quality for French to English."""
    # Check model availability first
    models_res = await client.get("/api/models")
    available_models = models_res.json()["models"]
    if not any(
        m["src_lang"] == "fr" and m["tgt_lang"] == "en" for m in available_models
    ):
        pytest.skip("fr-en model needed for this test")

    sources = [p[0] for p in FR_EN_PAIRS]
    refs = [[p[1] for p in FR_EN_PAIRS]]

    hyps = await translate_batch(client, sources, "fr", "en")
    assert len(hyps) == len(sources)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    chrf = sacrebleu.corpus_chrf(hyps, refs)

    print(f"\nFR->EN Quality: BLEU={bleu.score:.2f}, CHRF={chrf.score:.2f}")

    # Assert minimum quality
    assert bleu.score > 40.0
    assert chrf.score > 70.0
