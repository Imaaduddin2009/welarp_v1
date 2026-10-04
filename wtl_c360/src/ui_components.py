"""Small shared Streamlit presentation helpers for the demo app."""
from __future__ import annotations

import inspect
import re
from collections.abc import MutableMapping

import pandas as pd
import streamlit as st


def reset_role_state(state: MutableMapping[str, object]) -> None:
    """Clear role-specific answers and selections when the active identity changes."""
    state["messages"] = []
    state["last_response"] = None
    state["selected_customer_id"] = None
    state.pop("pending_question", None)
    state.pop("selected_demo_customer", None)
    state.pop("queue_table", None)


def readable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation copy with readable labels and normalized text."""
    result = frame.copy()
    result.columns = [str(column).replace("_", " ").strip().title() for column in result.columns]
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value: re.sub(r"\s+", " ", value).strip()
            if isinstance(value, str) else value
        )
    return result


def show_table(
    frame: pd.DataFrame,
    *,
    column_config: dict[str, object] | None = None,
    key: str | None = None,
    on_select: str = "ignore",
    selection_mode: str = "multi-row",
) -> object:
    """Render structured data, never a CSV/text serialization."""
    width_arg = (
        {"width": "stretch"}
        if "width" in inspect.signature(st.dataframe).parameters
        else {"use_container_width": True}
    )
    return st.dataframe(
        readable_frame(frame),
        hide_index=True,
        column_config=column_config,
        key=key,
        on_select=on_select,
        selection_mode=selection_mode,
        **width_arg,
    )


def show_sql(sql: str) -> None:
    """Show a query in a default-open expander, wrapping when supported."""
    with st.expander("Query used", expanded=True):
        if "wrap_lines" in inspect.signature(st.code).parameters:
            st.code(sql, language="sql", wrap_lines=True)
        else:
            st.code(sql, language="sql")


def refusal_suggestion(response: dict[str, object]) -> str | None:
    """Extract the unchanged suggestion text emitted by the Layer 2 refusal."""
    match = re.search(r'Try instead: "([^"]+)"', str(response.get("answer", "")))
    return match.group(1) if match else None


def show_refusal(response: dict[str, object], key: str) -> None:
    """Render a Layer 2 refusal as a separate callout with a usable suggestion."""
    st.warning(str(response.get("answer", "This request was refused.")))
    if response.get("reason"):
        st.caption(f"Reason: {response['reason']}")
    suggestion = refusal_suggestion(response)
    if suggestion and st.button(f"Try: {suggestion}", key=key):
        st.session_state["pending_question"] = suggestion


def show_qa_response(response: dict[str, object], key: str) -> None:
    """Render a structured, auditable Layer 2 response."""
    if response.get("refused"):
        show_refusal(response, key)
        return

    st.markdown(f"**{response.get('answer', '')}**")
    table = response.get("table")
    row_count = len(table) if isinstance(table, pd.DataFrame) else 0
    st.caption(
        f"Route: {response.get('route', 'unknown')} · "
        f"Freshness: {response.get('freshness', 'unknown')} · "
        f"Rows: {row_count:,}"
    )
    source = str(response.get("sql_or_sources", "")).strip()
    if source:
        if response.get("route") == "sql":
            show_sql(source)
        else:
            st.caption(f"Cited note/call/policy IDs: {source}")
    if isinstance(table, pd.DataFrame) and not table.empty:
        table = table.copy()
        if response.get("route") == "text" and {"source_type", "text"}.issubset(table.columns):
            from layer2 import CUES

            table["hardship_cue_badge"] = table.apply(
                lambda row: (
                    "Human review"
                    if row["source_type"] != "policy" and re.search(CUES, str(row["text"]), re.I)
                    else "—"
                ),
                axis=1,
            )
        config = {}
        for column in table.columns:
            lowered = str(column).lower()
            if "amount" in lowered or "balance" in lowered:
                config[str(column).replace("_", " ").title()] = st.column_config.NumberColumn(
                    format="$%.2f"
                )
            elif "rate" in lowered or "confidence" in lowered or "similarity" in lowered:
                config[str(column).replace("_", " ").title()] = st.column_config.NumberColumn(
                    format="%.1f%%" if "pct" in lowered else "%.3f"
                )
            elif "hardship_cue_badge" in lowered:
                config[str(column).replace("_", " ").title()] = st.column_config.TextColumn(
                    "Hardship cue"
                )
        show_table(table, column_config=config or None)
