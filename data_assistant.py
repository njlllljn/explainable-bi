"""Small local dataset Q&A helper; it never sends data outside the app."""

from __future__ import annotations

import re
import pandas as pd


def answer_dataset_question(frame: pd.DataFrame, question: str) -> tuple[str, pd.DataFrame | None]:
    """Answer common descriptive questions about the loaded dataset.

    This deliberately uses transparent pandas summaries rather than an LLM.
    Returns a plain-language answer and an optional small table for display.
    """
    q = question.lower().strip()
    if not q:
        return "Type a question about the dataset to get started.", None

    if any(word in q for word in ["how many rows", "number of rows", "record count", "how many records", "dataset size"]):
        return f"The loaded dataset contains **{len(frame):,} rows** and **{len(frame.columns):,} columns**. A row is one record in your file.", None
    if any(word in q for word in ["which columns", "what columns", "column names", "fields", "variables"]):
        return "The dataset columns are: " + ", ".join(f"`{column}`" for column in frame.columns) + ".", None
    if any(word in q for word in ["missing", "blank", "null", "incomplete"]):
        missing = frame.isna().sum().rename("Missing cells").to_frame()
        missing["Missing %"] = (frame.isna().mean() * 100).round(1)
        missing = missing[missing["Missing cells"] > 0].sort_values("Missing cells", ascending=False)
        if missing.empty:
            return "There are no blank cells in the loaded dataset.", None
        return "These columns contain missing values. The ML workflows fill numeric feature blanks with the training median; this summary reports the original uploaded data.", missing.reset_index(names="Column")

    # Match a real column name, including natural-language variants.
    normalized = {re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip(): c for c in frame.columns}
    found = None
    cleaned_question = re.sub(r"[^a-z0-9]+", " ", q).strip()
    group_phrase = next((phrase for phrase in [" by ", " for "] if phrase in f" {cleaned_question} "), None)
    column_search_text = cleaned_question.split(group_phrase, 1)[0].strip() if group_phrase else cleaned_question
    for name in sorted(normalized, key=len, reverse=True):
        if name and name in column_search_text:
            found = normalized[name]
            break
    if found is None:
        return ("I can answer questions about row counts, column names, missing values, averages, totals, medians, minimums, maximums, category counts, and correlations. "
                "Try: ‘What is the average revenue?’ or ‘Which columns have missing values?’"), None

    series = frame[found]
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if "correlat" in q or any(phrase in q for phrase in ["what drives", "related to", "relationship with", "moves with"]):
        numeric_columns = frame.select_dtypes(include="number").columns
        if found not in numeric_columns:
            return f"`{found}` is not numeric, so a Pearson correlation is not available. Ask for its category counts instead.", None
        corr = frame[numeric_columns].corr(numeric_only=True)[found].drop(found, errors="ignore").dropna().sort_values(key=abs, ascending=False)
        if corr.empty:
            return f"I couldn't calculate correlations for `{found}` because there are no other usable numeric columns.", None
        table = corr.head(8).rename("Correlation with " + str(found)).reset_index(names="Column")
        strongest = table.iloc[0]
        direction = "move together" if strongest.iloc[1] > 0 else "move in opposite directions"
        return (f"The strongest numeric association with `{found}` is `{strongest['Column']}` (correlation {strongest.iloc[1]:.2f}); the two measures tend to {direction}. "
                "Correlation describes association, not cause and effect."), table

    # Support grouped questions such as "average revenue by risk".
    if group_phrase:
        group_match = None
        for name in sorted(normalized, key=len, reverse=True):
            if name != re.sub(r"[^a-z0-9]+", " ", str(found).lower()).strip() and name in cleaned_question:
                group_match = normalized[name]
                break
        if group_match is not None:
            grouped = frame.assign(_question_value=numeric).groupby(group_match, dropna=False)["_question_value"]
            if "median" in q:
                grouped_result = grouped.median()
                measure = "median"
            elif any(word in q for word in ["total", "sum"]):
                grouped_result = grouped.sum()
                measure = "total"
            else:
                grouped_result = grouped.mean()
                measure = "average"
            table = grouped_result.rename(f"{measure.title()} {found}").reset_index()
            return f"Here is the {measure} `{found}` for each value of `{group_match}`. This describes groups in the uploaded data; it does not explain what caused differences.", table

    if len(valid) and (any(word in q for word in ["average", "mean", "typical"]) or re.search(r"\bavg\b", q)):
        return f"The average `{found}` is **{valid.mean():,.3f}**, calculated from {len(valid):,} non-missing numeric values.", None
    if len(valid) and any(word in q for word in ["median"]):
        return f"The median `{found}` is **{valid.median():,.3f}** across {len(valid):,} non-missing numeric values. Half the observed values are below it and half above it.", None
    if len(valid) and any(word in q for word in ["total", "sum"]):
        return f"The total `{found}` is **{valid.sum():,.3f}**, summed across {len(valid):,} non-missing numeric values.", None
    if len(valid) and any(word in q for word in ["highest", "maximum", "max", "largest"]):
        return f"The highest `{found}` is **{valid.max():,.3f}**.", None
    if len(valid) and any(word in q for word in ["lowest", "minimum", "min", "smallest"]):
        return f"The lowest `{found}` is **{valid.min():,.3f}**.", None
    if any(word in q for word in ["count", "how many", "distribution", "breakdown", "categories", "most common"]):
        counts = series.astype("string").fillna("(blank)").value_counts().head(12).rename("Records").to_frame()
        return f"Here are the most common values in `{found}` (up to 12).", counts.reset_index(names="Value")
    if len(valid):
        return f"`{found}` is numeric. Ask for its average, median, total, highest value, lowest value, distribution, or correlation to explore it.", None
    counts = series.astype("string").fillna("(blank)").value_counts().head(12).rename("Records").to_frame()
    return f"`{found}` is categorical text. Here are the most common values.", counts.reset_index(names="Value")
