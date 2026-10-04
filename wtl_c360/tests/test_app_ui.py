from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ui_components


def test_role_switch_reset_clears_customer_and_answer_state():
    state = {
        "messages": [{"role": "assistant", "content": "stale answer"}],
        "last_response": {"answer": "stale"},
        "selected_customer_id": "stale-customer",
        "pending_question": "stale suggestion",
        "selected_demo_customer": "stale example",
        "queue_table": {"selection": {"rows": [0]}},
    }

    ui_components.reset_role_state(state)

    assert state["messages"] == []
    assert state["last_response"] is None
    assert state["selected_customer_id"] is None
    assert "pending_question" not in state
    assert "selected_demo_customer" not in state
    assert "queue_table" not in state


def test_refusal_display_includes_reason_and_clickable_suggestion(monkeypatch):
    warnings: list[str] = []
    captions: list[str] = []
    buttons: list[str] = []
    monkeypatch.setattr(ui_components.st, "warning", warnings.append)
    monkeypatch.setattr(ui_components.st, "caption", captions.append)
    monkeypatch.setattr(
        ui_components.st,
        "button",
        lambda label, **kwargs: buttons.append(label) or False,
    )
    response = {
        "answer": 'I can\'t answer that: personal data is masked. '
        'Try instead: "How many customers consented to phone contact?"',
        "reason": "personal data is masked",
        "refused": True,
    }

    ui_components.show_refusal(response, "refusal-test")

    assert warnings and "I can't answer that" in warnings[0]
    assert any("personal data is masked" in caption for caption in captions)
    assert buttons == ['Try: How many customers consented to phone contact?']


def test_readable_frame_removes_carriage_returns_and_titles_columns():
    result = ui_components.readable_frame(
        pd.DataFrame({"note_text_masked": ["masked\r\ntext"], "total_overdue": [12.5]})
    )

    assert result.columns.tolist() == ["Note Text Masked", "Total Overdue"]
    assert result.iloc[0, 0] == "masked text"


def test_app_imports_and_qa_stays_offline_with_an_api_key_present():
    environment = os.environ.copy()
    environment["ANTHROPIC_API_KEY"] = "local-test-value"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app; assert app.qa_client('analyst', 'test', '').use_llm is False",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
