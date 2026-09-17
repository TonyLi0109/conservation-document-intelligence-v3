"""The deployed model selector stays intentionally narrow and deterministic."""
from config import CHAT_MODEL_OPTIONS, ModelSettings, RetrievalSettings


def test_only_approved_generation_models_are_exposed():
    assert CHAT_MODEL_OPTIONS == ("gpt-5.6-sol", "gpt-4.1-mini")


def test_sol_is_the_default_generation_model(monkeypatch):
    monkeypatch.delenv("V3_LLM_MODEL", raising=False)
    assert ModelSettings().llm_model == "gpt-5.6-sol"
    assert CHAT_MODEL_OPTIONS[0] == ModelSettings().llm_model


def test_environment_can_select_the_only_alternative(monkeypatch):
    monkeypatch.setenv("V3_LLM_MODEL", "gpt-4.1-mini")
    assert ModelSettings().llm_model == "gpt-4.1-mini"



def test_experimental_baseline_retrieval_depth_is_six(monkeypatch):
    monkeypatch.delenv("V3_TOP_K", raising=False)
    assert RetrievalSettings().top_k == 6
