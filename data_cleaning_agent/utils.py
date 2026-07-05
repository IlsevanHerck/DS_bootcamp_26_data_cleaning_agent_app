# Utility functions for lightweight data cleaning agent

import re
import logging
import warnings
from typing import Optional, Set

import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
    
    summary = f"""
        Dataset Summary:
        ----------------
        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}"""

    return summary.strip()


def get_input_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column summary of a DataFrame for display before cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.

    Returns
    -------
    pd.DataFrame
        Summary with column name, datatype, average, min, max, and fill rate.
    """
    total_rows = len(df)
    rows = []

    for col in df.columns:
        series = df[col]
        filled_pct = (series.notna().sum() / total_rows * 100) if total_rows else 0.0

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            average = non_null.mean() if len(non_null) else None
            minimum = non_null.min() if len(non_null) else None
            maximum = non_null.max() if len(non_null) else None
        elif pd.api.types.is_datetime64_any_dtype(series):
            non_null = series.dropna()
            average = None
            minimum = non_null.min() if len(non_null) else None
            maximum = non_null.max() if len(non_null) else None
        else:
            average = minimum = maximum = None

        rows.append(
            {
                "Column": col,
                "Datatype": str(series.dtype),
                "Average": average,
                "Minimum": minimum,
                "Maximum": maximum,
                "% Filled": round(filled_pct, 2),
            }
        )

    return pd.DataFrame(rows)


CLEANING_OPTION_INSTRUCTIONS = {
    "remove_high_missing_columns": (
        "Remove columns with more than 40% missing values"
    ),
    "impute_missing_values": (
        "Impute missing values (mean for numeric columns, mode for categorical columns)"
    ),
    "remove_duplicates": "Remove duplicate rows",
    "standardize_text": (
        "Standardize text columns (strip whitespace and apply consistent formatting)"
    ),
    "convert_data_types": "Convert columns to appropriate data types where possible",
    "remove_outliers": (
        "Remove statistical outliers from numeric columns using the IQR method"
    ),
}


def _has_convertible_object_columns(df: pd.DataFrame) -> bool:
    """Return True if any object column looks numeric or datetime-like."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col].dropna().astype(str).str.strip()
        if series.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            return True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed_dates = pd.to_datetime(series, errors="coerce")
        if parsed_dates.notna().mean() >= 0.8:
            return True
    return False


def get_cleaning_options(df: pd.DataFrame) -> list[dict]:
    """
    Determine which common cleaning steps apply to a DataFrame.

    Returns a list of option dicts with keys: id, label, description,
    applicable, default, and reason (when not applicable).
    """
    total_rows = len(df)
    missing_pct = (
        df.isna().sum() / total_rows * 100 if total_rows else pd.Series(dtype=float)
    )
    has_missing = bool(total_rows and missing_pct.gt(0).any())
    has_high_missing = bool(total_rows and missing_pct.gt(40).any())
    has_duplicates = bool(total_rows and df.duplicated().any())
    has_text = not df.select_dtypes(include=["object", "string"]).columns.empty
    has_numeric = not df.select_dtypes(include="number").columns.empty
    has_convertible = _has_convertible_object_columns(df)

    return [
        {
            "id": "remove_high_missing_columns",
            "label": "Remove columns with high missing values",
            "description": "Drop columns where more than 40% of values are missing",
            "applicable": has_high_missing,
            "default": True,
            "reason": "No columns exceed 40% missing values",
        },
        {
            "id": "impute_missing_values",
            "label": "Impute missing values",
            "description": "Fill missing values using mean (numeric) or mode (categorical)",
            "applicable": has_missing,
            "default": True,
            "reason": "No missing values in the dataset",
        },
        {
            "id": "remove_duplicates",
            "label": "Remove duplicate rows",
            "description": "Drop rows that are exact duplicates of other rows",
            "applicable": has_duplicates,
            "default": True,
            "reason": "No duplicate rows found",
        },
        {
            "id": "standardize_text",
            "label": "Standardize text formatting",
            "description": "Strip whitespace and normalize text in string columns",
            "applicable": has_text,
            "default": True,
            "reason": "No text columns in the dataset",
        },
        {
            "id": "convert_data_types",
            "label": "Convert data types",
            "description": "Convert columns to numeric or datetime types where appropriate",
            "applicable": has_convertible,
            "default": True,
            "reason": "No columns appear to need type conversion",
        },
        {
            "id": "remove_outliers",
            "label": "Remove outliers",
            "description": "Remove statistical outliers from numeric columns (IQR method)",
            "applicable": has_numeric,
            "default": False,
            "reason": "No numeric columns in the dataset",
        },
    ]


def build_cleaning_instructions(selected_option_ids: list[str]) -> str:
    """Build user instructions for the agent from selected cleaning option ids."""
    if not selected_option_ids:
        return "Do not apply any cleaning steps. Return the data unchanged."

    lines = [
        CLEANING_OPTION_INSTRUCTIONS[option_id]
        for option_id in selected_option_ids
        if option_id in CLEANING_OPTION_INSTRUCTIONS
    ]
    return "\n".join(f"- {line}" for line in lines)


CLEANING_OPTION_LABELS = {
    "remove_high_missing_columns": "Remove columns with high missing values",
    "impute_missing_values": "Impute missing values",
    "remove_duplicates": "Remove duplicate rows",
    "standardize_text": "Standardize text formatting",
    "convert_data_types": "Convert data types",
    "remove_outliers": "Remove outliers",
}

# Minimum sample sizes for reliable application of each cleaning method.
MIN_ROWS_ANY_CLEANING = 10
MIN_ROWS_MISSING_RATE = 20
MIN_ROWS_IMPUTATION = 30
MIN_NON_NULL_IMPUTATION = 5
MIN_ROWS_DEDUP = 15
MIN_ROWS_TEXT = 10
MIN_ROWS_TYPE_CONVERSION = 15
MIN_NON_NULL_TYPE_CONVERSION = 5
MIN_ROWS_OUTLIERS = 30
HIGH_MISSING_THRESHOLD_PCT = 40


def _columns_with_high_missing(df: pd.DataFrame) -> Set[str]:
    """Return columns that exceed the high-missing threshold."""
    row_count = len(df)
    if not row_count:
        return set()

    return {
        col
        for col in df.columns
        if df[col].isna().sum() / row_count * 100 > HIGH_MISSING_THRESHOLD_PCT
    }


def get_cleaning_size_warnings(
    df: pd.DataFrame, selected_option_ids: list[str]
) -> list[dict]:
    """
    Return warnings when the dataset may be too small for selected cleaning steps.

    Each warning includes option_id, label, message, and min_recommended rows.
    """
    if not selected_option_ids:
        return []

    warnings = []
    row_count = len(df)
    drop_high_missing_selected = "remove_high_missing_columns" in selected_option_ids
    columns_to_be_dropped = (
        _columns_with_high_missing(df) if drop_high_missing_selected else set()
    )

    if row_count < MIN_ROWS_ANY_CLEANING:
        warnings.append(
            {
                "option_id": None,
                "label": "Dataset size",
                "message": (
                    f"The dataset has only {row_count} rows. Cleaning results are "
                    f"generally unreliable below {MIN_ROWS_ANY_CLEANING} rows."
                ),
                "min_recommended": MIN_ROWS_ANY_CLEANING,
            }
        )

    for option_id in selected_option_ids:
        label = CLEANING_OPTION_LABELS.get(option_id, option_id)

        if option_id == "remove_high_missing_columns":
            if row_count < MIN_ROWS_MISSING_RATE:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Missing-value rates need at least "
                            f"{MIN_ROWS_MISSING_RATE} rows to be stable "
                            f"(current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_MISSING_RATE,
                    }
                )

        elif option_id == "impute_missing_values":
            if row_count < MIN_ROWS_IMPUTATION:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Mean and mode imputation are unreliable below "
                            f"{MIN_ROWS_IMPUTATION} rows (current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_IMPUTATION,
                    }
                )

            sparse_columns = [
                col
                for col in df.columns
                if col not in columns_to_be_dropped
                and df[col].isna().any()
                and df[col].notna().sum() < MIN_NON_NULL_IMPUTATION
            ]
            if sparse_columns:
                cols = ", ".join(sparse_columns)
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Column(s) {cols} have fewer than "
                            f"{MIN_NON_NULL_IMPUTATION} observed values to impute from."
                        ),
                        "min_recommended": MIN_NON_NULL_IMPUTATION,
                    }
                )

        elif option_id == "remove_duplicates":
            duplicate_count = int(df.duplicated().sum())
            if row_count < MIN_ROWS_DEDUP:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Removing duplicates from fewer than {MIN_ROWS_DEDUP} rows "
                            f"can remove too much of the dataset (current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_DEDUP,
                    }
                )
            elif (
                duplicate_count > 0
                and row_count < MIN_ROWS_IMPUTATION
                and duplicate_count / row_count > 0.2
            ):
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Removing {duplicate_count} duplicate row(s) would drop "
                            f"more than 20% of this {row_count}-row dataset."
                        ),
                        "min_recommended": MIN_ROWS_IMPUTATION,
                    }
                )

        elif option_id == "standardize_text":
            if row_count < MIN_ROWS_TEXT:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Text standardization has limited value below "
                            f"{MIN_ROWS_TEXT} rows (current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_TEXT,
                    }
                )

        elif option_id == "convert_data_types":
            if row_count < MIN_ROWS_TYPE_CONVERSION:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Type conversion needs at least "
                            f"{MIN_ROWS_TYPE_CONVERSION} rows to validate reliably "
                            f"(current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_TYPE_CONVERSION,
                    }
                )

            sparse_columns = [
                col
                for col in df.select_dtypes(include=["object", "string"]).columns
                if df[col].notna().sum() < MIN_NON_NULL_TYPE_CONVERSION
            ]
            if sparse_columns:
                cols = ", ".join(sparse_columns)
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"Column(s) {cols} have fewer than "
                            f"{MIN_NON_NULL_TYPE_CONVERSION} values to infer a type from."
                        ),
                        "min_recommended": MIN_NON_NULL_TYPE_CONVERSION,
                    }
                )

        elif option_id == "remove_outliers":
            if row_count < MIN_ROWS_OUTLIERS:
                warnings.append(
                    {
                        "option_id": option_id,
                        "label": label,
                        "message": (
                            f"IQR-based outlier detection needs at least "
                            f"{MIN_ROWS_OUTLIERS} rows for stable quartiles "
                            f"(current: {row_count})."
                        ),
                        "min_recommended": MIN_ROWS_OUTLIERS,
                    }
                )

    return warnings


def _detect_high_missing_column_removal(
    df_raw: pd.DataFrame, df_cleaned: pd.DataFrame
) -> tuple[bool, str]:
    dropped_cols = set(df_raw.columns) - set(df_cleaned.columns)
    if not dropped_cols:
        return False, "No columns were removed"

    total_rows = len(df_raw)
    high_missing_cols = [
        col
        for col in df_raw.columns
        if total_rows and df_raw[col].isna().sum() / total_rows * 100 > 40
    ]
    removed_high_missing = sorted(set(high_missing_cols) & dropped_cols)
    if removed_high_missing:
        cols = ", ".join(removed_high_missing)
        return True, f"Removed {len(removed_high_missing)} column(s): {cols}"

    cols = ", ".join(sorted(dropped_cols))
    return True, f"Removed {len(dropped_cols)} column(s): {cols}"


def _detect_imputation(df_raw: pd.DataFrame, df_cleaned: pd.DataFrame) -> tuple[bool, str]:
    common_cols = [col for col in df_raw.columns if col in df_cleaned.columns]
    raw_missing = int(df_raw[common_cols].isna().sum().sum()) if common_cols else 0
    clean_missing = int(df_cleaned[common_cols].isna().sum().sum()) if common_cols else 0
    filled = raw_missing - clean_missing
    if filled > 0:
        return True, f"Filled {filled} missing value(s)"
    return False, "No missing values were filled"


def _detect_duplicate_removal(
    df_raw: pd.DataFrame, df_cleaned: pd.DataFrame
) -> tuple[bool, str]:
    duplicates_in_raw = int(df_raw.duplicated().sum())
    rows_removed = len(df_raw) - len(df_cleaned)
    if duplicates_in_raw == 0:
        return False, "No duplicate rows were found in the input data"
    if rows_removed <= 0:
        return False, "No rows were removed"

    duplicates_in_cleaned = int(df_cleaned.duplicated().sum())
    if duplicates_in_cleaned == 0 and rows_removed > 0:
        return True, f"Removed {rows_removed} duplicate row(s)"

    return False, f"{rows_removed} row(s) removed, but duplicates may still remain"


def _detect_text_standardization(
    df_raw: pd.DataFrame, df_cleaned: pd.DataFrame
) -> tuple[bool, str]:
    changed_columns = []
    for col in df_raw.columns:
        if col not in df_cleaned.columns:
            continue
        if not (
            pd.api.types.is_object_dtype(df_raw[col])
            or pd.api.types.is_string_dtype(df_raw[col])
        ):
            continue

        raw_series = df_raw[col].dropna().astype(str)
        clean_series = df_cleaned[col].dropna().astype(str)
        if raw_series.empty:
            continue

        whitespace_issues = (raw_series != raw_series.str.strip()).sum()
        if whitespace_issues == 0:
            continue

        clean_values = set(clean_series.str.strip())
        normalized = sum(1 for value in raw_series if value.strip() in clean_values)
        if normalized > 0:
            changed_columns.append(col)

    if changed_columns:
        cols = ", ".join(changed_columns)
        return True, f"Standardized text in column(s): {cols}"

    return False, "No text formatting changes detected"


def _detect_type_conversion(
    df_raw: pd.DataFrame, df_cleaned: pd.DataFrame
) -> tuple[bool, str]:
    converted_columns = []
    for col in df_raw.columns:
        if col not in df_cleaned.columns:
            continue
        if str(df_raw[col].dtype) != str(df_cleaned[col].dtype):
            converted_columns.append(
                f"{col} ({df_raw[col].dtype} -> {df_cleaned[col].dtype})"
            )

    if converted_columns:
        return True, "; ".join(converted_columns)

    return False, "No column data types were changed"


def _detect_outlier_removal(
    df_raw: pd.DataFrame, df_cleaned: pd.DataFrame
) -> tuple[bool, str]:
    rows_removed = len(df_raw) - len(df_cleaned)
    duplicates_in_raw = int(df_raw.duplicated().sum())
    if rows_removed <= 0:
        return False, "No rows were removed"

    numeric_cols = [
        col
        for col in df_raw.columns
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_raw[col])
    ]
    if not numeric_cols:
        return False, "No numeric columns available to assess outlier removal"

    range_changed = any(
        df_cleaned[col].min() > df_raw[col].min()
        or df_cleaned[col].max() < df_raw[col].max()
        for col in numeric_cols
    )
    unexplained_row_drop = rows_removed > duplicates_in_raw

    if range_changed and unexplained_row_drop:
        return True, f"Removed {rows_removed} row(s); numeric ranges were narrowed"

    if unexplained_row_drop:
        return True, f"Removed {rows_removed} row(s) beyond duplicate removal"

    return False, "No outlier-related row removals detected"


_CLEANING_STEP_DETECTORS = {
    "remove_high_missing_columns": _detect_high_missing_column_removal,
    "impute_missing_values": _detect_imputation,
    "remove_duplicates": _detect_duplicate_removal,
    "standardize_text": _detect_text_standardization,
    "convert_data_types": _detect_type_conversion,
    "remove_outliers": _detect_outlier_removal,
}


def get_applied_cleaning_summary(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    selected_option_ids: list[str],
) -> pd.DataFrame:
    """
    Summarize which selected cleaning steps appear to have been applied.

    Compares the raw and cleaned DataFrames to infer the effect of each step.
    """
    rows = []

    if not selected_option_ids:
        rows.append(
            {
                "Step": "No cleaning steps selected",
                "Status": "Not applied",
                "Details": "The dataset was returned without requested cleaning",
            }
        )
        return pd.DataFrame(rows)

    for option_id in selected_option_ids:
        label = CLEANING_OPTION_LABELS.get(option_id, option_id)
        detector = _CLEANING_STEP_DETECTORS.get(option_id)
        if detector is None:
            rows.append(
                {
                    "Step": label,
                    "Status": "Unknown",
                    "Details": "No detector available for this step",
                }
            )
            continue

        applied, details = detector(df_raw, df_cleaned)
        rows.append(
            {
                "Step": label,
                "Status": "Applied" if applied else "Not applied",
                "Details": details,
            }
        )

    return pd.DataFrame(rows)


def format_agent_log(user_instructions: Optional[str], response: Optional[dict]) -> str:
    """
    Format the agent run details into a readable log string.

    Parameters
    ----------
    user_instructions : str or None
        Cleaning instructions passed to the agent.
    response : dict or None
        Full workflow response from the agent.

    Returns
    -------
    str
        Formatted log text including instructions, metadata, and generated code.
    """
    if not response:
        return "No agent log available."

    sections = [
        "Data Cleaning Agent Log",
        "=" * 40,
        "",
        "User Instructions:",
        user_instructions or "None",
        "",
        f"Retry attempts: {response.get('retry_count', 0)}",
    ]

    error = response.get("data_cleaner_error")
    if error:
        sections.extend(["", f"Last error: {error}"])

    log_path = response.get("data_cleaner_function_path")
    if log_path:
        sections.extend(["", f"Saved to: {log_path}"])

    sections.extend(["", "Generated cleaning code:", "-" * 40, ""])
    sections.append(response.get("data_cleaner_function") or "No code generated.")

    return "\n".join(sections)


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    # Run the function and handle errors
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"
    
    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }
