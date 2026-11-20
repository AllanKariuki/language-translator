import pytest

from engine import MAX_CHARS, Translator, split_sentences


class FakeTranslator(Translator):
    def __init__(self):
        super().__init__("models/fake", backend="hf")
        self.calls = []

    def _load(self):
        self.tokenizer = None

    def _translate_raw(self, texts, src, tgt):
        self.calls.append((list(texts), src, tgt))
        return [f"<{t}>" for t in texts]


def test_split_single_sentence():
    assert split_sentences("Hello world") == [["Hello world"]]


def test_split_three_sentences():
    assert split_sentences("Hi. How are you? Fine!") == [["Hi.", "How are you?", "Fine!"]]


def test_split_multiline_with_blank_line():
    assert split_sentences("Hello.\n\nWorld!") == [["Hello."], [], ["World!"]]


def test_translate_empty_returns_empty_without_backend():
    t = FakeTranslator()
    assert t.translate("", "eng_Latn", "swh_Latn") == ""
    assert t.translate("   \n  ", "eng_Latn", "swh_Latn") == ""
    assert t.calls == []


def test_translate_one_call_and_reassembled():
    t = FakeTranslator()
    out = t.translate("Hi. How are you?\n\nFine!", "eng_Latn", "swh_Latn")
    assert len(t.calls) == 1
    texts, src, tgt = t.calls[0]
    assert texts == ["Hi.", "How are you?", "Fine!"]
    assert src == "eng_Latn"
    assert tgt == "swh_Latn"
    assert out == "<Hi.> <How are you?>\n\n<Fine!>"


def test_translate_unsupported_language():
    t = FakeTranslator()
    with pytest.raises(ValueError):
        t.translate("Hi.", "xx_Latn", "swh_Latn")


def test_translate_same_language():
    t = FakeTranslator()
    with pytest.raises(ValueError):
        t.translate("Hi.", "eng_Latn", "eng_Latn")


def test_translate_too_long():
    t = FakeTranslator()
    with pytest.raises(ValueError):
        t.translate("x" * (MAX_CHARS + 1), "eng_Latn", "swh_Latn")
