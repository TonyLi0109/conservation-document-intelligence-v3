"""The default test suite must never incur OpenAI API cost."""

import pytest


@pytest.fixture(autouse=True)
def forbid_paid_provider_calls(monkeypatch):
    import api_clients
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must use provider fixtures; run opt-in live evaluation separately")
    monkeypatch.setattr(api_clients, "_client", forbidden)
