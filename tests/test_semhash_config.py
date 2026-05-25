from unittest.mock import MagicMock, patch

import pytest

from kg_gen.kg_gen import KGGen, _env_semhash_model
from kg_gen.models import Graph
from kg_gen.utils.deduplicate import (
    DeduplicateList,
    _DEFAULT_SEMHASH_MODEL,
    _load_semhash_encoder,
    run_semhash_deduplication,
)


@pytest.fixture
def mock_encoder():
    encoder = MagicMock(name="encoder")
    encoder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return encoder


def test_singularize_skipped_for_cjk():
    dedup = DeduplicateList(semhash_encoder=MagicMock())
    text = "北京北京市"
    assert dedup.singularize(text) == text


def test_singularize_english_still_applies(mock_encoder):
    dedup = DeduplicateList(semhash_encoder=mock_encoder)
    assert dedup.singularize("cats") == "cat"


@patch("model2vec.StaticModel")
def test_load_semhash_encoder_default(mock_static_model, mock_encoder):
    mock_static_model.from_pretrained.return_value = mock_encoder

    result = _load_semhash_encoder(None)

    mock_static_model.from_pretrained.assert_called_once_with(_DEFAULT_SEMHASH_MODEL)
    assert result is mock_encoder


@patch("model2vec.StaticModel")
def test_load_semhash_encoder_custom(mock_static_model, mock_encoder):
    mock_static_model.from_pretrained.return_value = mock_encoder

    _load_semhash_encoder("minishlab/potion-multilingual-128M")

    mock_static_model.from_pretrained.assert_called_once_with(
        "minishlab/potion-multilingual-128M"
    )


@patch("kg_gen.utils.deduplicate.SemHash")
@patch("kg_gen.utils.deduplicate._load_semhash_encoder")
def test_semhash_model_passed_to_from_records(
    mock_load_encoder, mock_semhash, mock_encoder
):
    mock_load_encoder.return_value = mock_encoder
    mock_instance = MagicMock()
    mock_semhash.from_records.return_value = mock_instance
    mock_instance.self_deduplicate.return_value = MagicMock(
        selected=["cat"],
        duplicates=[],
    )

    dedup = DeduplicateList(semhash_model="minishlab/potion-multilingual-128M")
    dedup.deduplicate(["cats"])

    mock_load_encoder.assert_called_once_with("minishlab/potion-multilingual-128M")
    mock_semhash.from_records.assert_called_once_with(
        records=["cat"],
        model=mock_encoder,
    )


@patch("kg_gen.utils.deduplicate.SemHash")
@patch("kg_gen.utils.deduplicate._load_semhash_encoder")
def test_run_semhash_loads_model_once(mock_load_encoder, mock_semhash, mock_encoder):
    mock_load_encoder.return_value = mock_encoder
    mock_instance = MagicMock()
    mock_semhash.from_records.return_value = mock_instance
    mock_instance.self_deduplicate.return_value = MagicMock(
        selected=[],
        duplicates=[],
    )

    graph = Graph(
        entities=["北京"], edges=["位于"], relations=[["北京", "位于", "中国"]]
    )
    run_semhash_deduplication(
        graph,
        semhash_model="minishlab/potion-multilingual-128M",
    )

    mock_load_encoder.assert_called_once_with("minishlab/potion-multilingual-128M")
    assert mock_semhash.from_records.call_count == 2
    for call in mock_semhash.from_records.call_args_list:
        assert call.kwargs["model"] is mock_encoder


def test_env_semhash_model_default(monkeypatch):
    monkeypatch.delenv("SEMHASH_MODEL", raising=False)
    assert _env_semhash_model() == "minishlab/potion-multilingual-128M"


def test_env_semhash_model_from_env(monkeypatch):
    monkeypatch.setenv("SEMHASH_MODEL", "minishlab/potion-base-8M")
    assert _env_semhash_model() == "minishlab/potion-base-8M"


def test_kggen_env_semhash_model_default(monkeypatch):
    monkeypatch.delenv("SEMHASH_MODEL", raising=False)
    kg = KGGen()
    assert kg.semhash_model == "minishlab/potion-multilingual-128M"


def test_kggen_semhash_model_override(monkeypatch):
    monkeypatch.setenv("SEMHASH_MODEL", "minishlab/potion-multilingual-128M")
    kg = KGGen(semhash_model="minishlab/potion-base-8M")
    assert kg.semhash_model == "minishlab/potion-base-8M"
