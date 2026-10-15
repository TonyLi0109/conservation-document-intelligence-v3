"""Offline synthetic walkthrough; scripted responses are NOT a model experiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_qgcg_groups import build_qgcg, configure_qgcg


def main():
    requirement = {
        "requirement": "Compare the totals in 2020 and 2025.",
        "why_indispensable": "The question explicitly asks for this comparison.",
    }
    pages = {
        "A": "In 2020 the total was 10; in 2025 it was 20.",
        "B": "The 2020 total was 10.",
        "C": "The 2025 total was 20.",
    }
    responses = {
        "model_a": [requirement],
        "model_b": {"mappings": [{
            **requirement,
            "evidence_options": [["A"], ["B", "C"]],
            "source_excerpts": [
                {"page_id": key, "quote": text, "explanation": "Supplies the stated year totals."}
                for key, text in pages.items()
            ],
            "status": "required",
            "issues": [],
        }]},
        "model_c": [],
    }

    def scripted_complete(*, role, system_prompt, payload, json_schema):
        return json.dumps(responses[role])

    configure_qgcg(scripted_complete, max_repair_rounds=1)
    result = build_qgcg(
        "Compare the 2020 and 2025 totals.",
        [{"page_id": key, "text": text} for key, text in pages.items()],
    )
    print(json.dumps({"demo_only": True, "artifact": result}, indent=2))


if __name__ == "__main__":
    main()
