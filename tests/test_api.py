import pytest
from fastapi.testclient import TestClient

from api import create_app
from engine import Translator
from feedback import FeedbackStore


class FakeTranslator(Translator):
    def __init__(self):
        super().__init__("models/fake", backend="hf")

    def _load(self):
        self.tokenizer = None

    def _translate_raw(self, texts, src, tgt):
        return [f"<{t}>" for t in texts]


@pytest.fixture
def client(tmp_path):
    app = create_app(translator=FakeTranslator(), store=FeedbackStore(str(tmp_path / "f.db")))
    with TestClient(app) as client:
        yield client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "hf", "model": "fake"}


def test_languages(client):
    r = client.get("/languages")
    assert r.status_code == 200
    assert r.json() == {"eng_Latn": "English", "swh_Latn": "Swahili"}


def test_translate_success(client):
    r = client.post("/translate", json={"text": "Hello.", "source_lang": "eng_Latn", "target_lang": "swh_Latn"})
    assert r.status_code == 200
    assert r.json() == {"translation": "<Hello.>", "source_lang": "eng_Latn", "target_lang": "swh_Latn"}


def test_translate_bad_code(client):
    r = client.post("/translate", json={"text": "Hi.", "source_lang": "xx_Latn", "target_lang": "swh_Latn"})
    assert r.status_code == 422


def test_translate_same_lang(client):
    r = client.post("/translate", json={"text": "Hi.", "source_lang": "eng_Latn", "target_lang": "eng_Latn"})
    assert r.status_code == 422


def test_feedback_success(client):
    r = client.post(
        "/feedback",
        json={
            "source_text": "Hello.",
            "source_lang": "eng_Latn",
            "target_lang": "swh_Latn",
            "translation": "Habari.",
            "rating": "up",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"id": 1}
    assert client.app.state.store.count() == 1


def test_feedback_neither_rating_nor_correction(client):
    r = client.post(
        "/feedback",
        json={
            "source_text": "Hello.",
            "source_lang": "eng_Latn",
            "target_lang": "swh_Latn",
            "translation": "Habari.",
        },
    )
    assert r.status_code == 422


def test_feedback_bad_rating(client):
    r = client.post(
        "/feedback",
        json={
            "source_text": "Hello.",
            "source_lang": "eng_Latn",
            "target_lang": "swh_Latn",
            "translation": "Habari.",
            "rating": "bad",
        },
    )
    assert r.status_code == 422
