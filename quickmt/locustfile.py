import random
from locust import FastHttpUser, task, between


class TranslationUser(FastHttpUser):
    wait_time = between(0, 0)

    # Sample sentences for translation and identification
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Can we translate this correctly and quickly?",
        "هذا نص تجريبي باللغة العربية.",  # Arabic
        "الذكاء الاصطناعي هو المستقبل.",  # Arabic (AI is the future)
        "أحب تعلم لغات جديدة.",  # Arabic (I love learning new languages)
        "这是一段中文测试文本。",  # Chinese
        "人工智能正在改变世界。",  # Chinese (AI is changing the world)
        "今天天气真好，去公园散步。",  # Chinese (Weather is nice, let's walk)
        "Bonjour, comment allez-vous ?",  # French
        "L'intelligence artificielle transforme notre vie quotidienne.",  # French (AI transforms daily life)
        "Ceci est un exemple de phrase en français.",  # French
    ]

    def on_start(self):
        """Discover available models on startup."""
        try:
            response = self.client.get("/models")
            if response.status_code == 200:
                self.available_models = response.json().get("models", [])
                if not self.available_models:
                    print("No models found. Load test might fail.")
            else:
                self.available_models = []
        except Exception as e:
            print(f"Error discovering models: {e}")
            self.available_models = []

    def get_random_model(self):
        """
        Return a model, favoring the first 3 (hot set) 99% of the time,
        and others (cold set) 1% of the time to trigger LRU eviction.
        """
        if not self.available_models:
            return None

        # If we have 4 or more models, we can simulate eviction cycles
        if len(self.available_models) >= 4:
            # 99.99% chance to pick from the first 3
            if random.random() < 0.9999:
                return random.choice(self.available_models[:3])
            else:
                # 0.01% chance to pick from the rest
                return random.choice(self.available_models[3:])

        return random.choice(self.available_models)

    @task(1)
    def translate_single(self):
        model = self.get_random_model()
        if not model:
            return

        self.client.post(
            "/translate",
            json={
                "src": random.choice(self.sample_texts) + str(random.random()),
                "src_lang": model["src_lang"],
                "tgt_lang": model["tgt_lang"],
                "beam_size": 2,
            },
            name="/translate [single, manual]",
        )

    @task(1)
    def translate_auto_detect(self):
        """Translate without specifying src_lang to trigger LangID."""
        ret = self.client.post(
            "/translate",
            json={
                "src": random.choice(self.sample_texts) + str(random.random()),
                "tgt_lang": "en",
                "beam_size": 2,
            },
            name="/translate [single, auto-detect]",
        )
        ret_json = ret.json()
        assert "src_lang" in ret_json
        assert "tgt_lang" in ret_json
        assert "translation" in ret_json
        assert "src_lang_score" in ret_json
        assert "model_used" in ret_json
        assert ret_json["tgt_lang"] == "en"

    @task(1)
    def translate_list(self):
        model = self.get_random_model()
        if not model:
            return

        num_sentences = random.randint(2, 5)
        texts = random.sample(self.sample_texts, num_sentences)
        texts = [i + str(random.random()) for i in texts]
        ret = self.client.post(
            "/translate",
            json={
                "src": texts,
                "src_lang": model["src_lang"],
                "tgt_lang": model["tgt_lang"],
                "beam_size": 2,
            },
            name="/translate [list, manual]",
        )
        ret_json = ret.json()
        for i in ret_json["src_lang"]:
            assert i == model["src_lang"]
        assert ret_json["tgt_lang"] == model["tgt_lang"]
        assert len(ret_json["translation"]) == num_sentences

    @task(1)
    def identify_language(self):
        """Directly benchmark the identification endpoint."""
        num_sentences = random.randint(1, 4)
        texts = random.sample(self.sample_texts, num_sentences)
        src = texts[0] if num_sentences == 1 else texts

        self.client.post(
            "/identify-language", json={"src": src}, name="/identify-language"
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")
