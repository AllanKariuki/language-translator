import pytest

from feedback import FeedbackStore


def test_add_with_rating(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    row_id = store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.", rating="up")
    assert row_id == 1
    assert store.count() == 1
    store.close()


def test_add_with_correction(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.", correction="Hujambo.")
    assert store.count() == 1
    store.close()


def test_add_without_rating_or_correction_raises(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    with pytest.raises(ValueError):
        store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.")
    store.close()


def test_add_with_bad_rating_raises(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    with pytest.raises(ValueError):
        store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.", rating="bad")
    store.close()


def test_blank_correction_without_rating_raises(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    with pytest.raises(ValueError):
        store.add("eng_Latn", "swh_Latn", "Hello.", "Habari.", correction="   ")
    store.close()


def test_count(tmp_path):
    store = FeedbackStore(str(tmp_path / "f.db"))
    assert store.count() == 0
    store.add("eng_Latn", "swh_Latn", "A", "B", rating="down")
    store.add("eng_Latn", "swh_Latn", "C", "D", correction="E")
    assert store.count() == 2
    store.close()
