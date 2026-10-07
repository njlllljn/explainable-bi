"""Append-only Excel audit log for user-approved what-if scenarios."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import pandas as pd


LOG_PATH = Path(__file__).parent / "outputs" / "decision_log.xlsx"


def append_decision(record: dict, path: Path = LOG_PATH) -> pd.DataFrame:
    """Append one scenario record and return the updated worksheet contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Logged at (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    row["Input changes (JSON)"] = json.dumps(row.get("Input changes", {}), ensure_ascii=False)
    row.pop("Input changes", None)
    previous = pd.read_excel(path, sheet_name="Decision log") if path.exists() else pd.DataFrame()
    updated = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
        updated.to_excel(writer, sheet_name="Decision log", index=False)
        sheet = writer.sheets["Decision log"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cells in sheet.columns:
            width = min(max(max(len(str(cell.value or "")) for cell in cells) + 2, 14), 48)
            sheet.column_dimensions[cells[0].column_letter].width = width
    return updated


def read_decisions(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name="Decision log")
