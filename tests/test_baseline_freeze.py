"""The V3 experimental freeze manifest remains synchronized with runtime defaults."""
import json
from pathlib import Path

from config import CHAT_MODEL_OPTIONS, ModelSettings, RetrievalSettings


ROOT = Path(__file__).resolve().parents[1]


def test_v3_baseline_manifest_matches_runtime_defaults():
    manifest = json.loads((ROOT / "v3_baseline_config.json").read_text(encoding="utf-8"))

    assert manifest["baseline_id"] == "CDIRP-V3-EXPERIMENTAL-BASELINE"
    assert manifest["git"]["tag"] == "v3.0-experimental-baseline"
    assert manifest["corpus"]["document_count"] == 36
    assert manifest["retrieval"]["production_mode"] == "hybrid_rerank"
    assert manifest["retrieval"]["top_k"] == RetrievalSettings().top_k == 6
    assert manifest["models"]["baseline_llm_model_id"] == ModelSettings().llm_model
    assert tuple(manifest["models"]["allowed_generation_models"]) == CHAT_MODEL_OPTIONS
    assert manifest["evaluation"]["top_k"] == 6
    assert manifest["evaluation"]["offline_failed_count"] == 0


def test_freeze_document_and_manifest_paths_exist():
    document = (ROOT / "V3_BASELINE_FREEZE.md").read_text(encoding="utf-8")
    assert "(?P<start>20\\d{2})" in document
    assert "(?P<end>20\\d{2})" in document
    assert (ROOT / "evaluation" / "baselines" / "offline.json").is_file()
