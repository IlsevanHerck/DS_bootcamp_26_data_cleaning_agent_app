# Data Cleaning Agent — Agent Instructions

AI-powered data cleaning app: users upload a CSV, review summaries, choose cleaning steps, and run an LLM agent that generates and executes pandas cleaning code.

## Project Overview

| Layer | Role |
|-------|------|
| `app.py` | Streamlit UI — upload, toggles, warnings, results, downloads |
| `data_cleaning_agent/data_cleaning_agent.py` | LangGraph agent — generate code, execute, retry on error |
| `data_cleaning_agent/utils.py` | Shared logic — summaries, options, warnings, step detection, logs |

Run the app: `poetry run streamlit run app.py`

## Streamlit User Flow

Implement and extend features to match this flow:

1. **Upload CSV** → load with `pd.read_csv`
2. **Input Data Summary** → `get_input_data_summary(df)` (column, datatype, average, min, max, % filled)
3. **Cleaning Options** → toggles from `get_cleaning_options(df)`; inapplicable options are disabled
4. **Clean Data** → `get_cleaning_size_warnings(df, selected_option_ids)`; if warnings exist, pause for user confirmation before running the agent
5. **Applied Cleaning Steps** → `get_applied_cleaning_summary(df_raw, df_cleaned, selected_option_ids)`
6. **Cleaned Data Summary** → `get_input_data_summary(df_cleaned)` + row preview
7. **Downloads** → cleaned CSV + agent log (hidden in expander via `format_agent_log`)

### Cleaning option IDs

Keep these in sync across `utils.py` and the UI:

| ID | Purpose |
|----|---------|
| `remove_high_missing_columns` | Drop columns with >40% missing |
| `impute_missing_values` | Mean (numeric) / mode (categorical) imputation |
| `remove_duplicates` | Drop exact duplicate rows |
| `standardize_text` | Strip/normalize text columns |
| `convert_data_types` | Convert object columns to numeric/datetime where appropriate |
| `remove_outliers` | IQR-based outlier removal (off by default) |

User selections are converted to agent instructions via `build_cleaning_instructions(selected_option_ids)` and passed as `user_instructions` to `LightweightDataCleaningAgent.invoke_agent`.

## Key Thresholds and Rules

Defined in `data_cleaning_agent/utils.py` — update together when changing behavior:

- `HIGH_MISSING_THRESHOLD_PCT = 40` — used for option applicability, agent instructions, and column-drop detection
- Size warnings use `MIN_ROWS_*` and `MIN_NON_NULL_*` constants per cleaning method
- **Imputation warnings**: do not warn about sparse columns that will be dropped when `remove_high_missing_columns` is also selected (use `_columns_with_high_missing`)

## Architecture Guidelines

### Separation of concerns

- **UI logic** → `app.py` (Streamlit widgets, session state, layout)
- **Business logic** → `data_cleaning_agent/utils.py` (summaries, applicability, warnings, post-clean detection)
- **Agent orchestration** → `data_cleaning_agent/data_cleaning_agent.py` (LangGraph workflow, LLM prompts, code execution)

Keep `app.py` thin. Prefer adding helpers in `utils.py` over embedding logic in Streamlit callbacks.

### Session state (`app.py`)

Used for the dataset-size confirmation gate:

- `awaiting_size_confirmation` — user must confirm or change settings
- `size_warnings` — list of warning dicts from `get_cleaning_size_warnings`
- `cleaning_signature` — `(filename, selected_option_ids)` to detect setting changes
- `execute_cleaning` — triggers the agent run after confirmation or when no warnings

Reset confirmation state when the user changes toggles or re-uploads a file.

### Agent backend

- Model: `ChatOpenAI(model="gpt-4o-mini", temperature=0)`
- Generated code saved to `logs/data_cleaner.py` when `log=True`
- Agent retries failed code up to 3 times via LangGraph
- Requires `OPENAI_API_KEY` in `.env` (never commit secrets)

## Adding a New Cleaning Option

Update all of the following:

1. `CLEANING_OPTION_INSTRUCTIONS` and `CLEANING_OPTION_LABELS` in `utils.py`
2. `get_cleaning_options()` — applicability detection
3. `get_cleaning_size_warnings()` — minimum sample size checks
4. `_CLEANING_STEP_DETECTORS` + detector function — post-clean “Applied” status
5. Agent prompt in `data_cleaning_agent.py` if default steps change
6. `README.md` — user-facing docs

The Streamlit toggle loop reads from `get_cleaning_options()` automatically; no hardcoded option list in `app.py`.

## Coding Standards

- **Python 3.9+** compatible (avoid `str | None`; use `Optional` from `typing`)
- **Dependencies**: manage with Poetry; run commands via `poetry run`
- **Minimal diffs**: match existing naming, types, and docstring style
- **pandas-first**: cleaning operations should use pandas/numpy, not bespoke row loops where vectorized ops exist
- **No over-engineering**: no extra abstractions for one-off helpers; no speculative error handling
- **Comments**: only for non-obvious business rules (e.g. why imputation skips columns slated for removal)
- **Tests**: add pytest tests for `utils.py` logic when behavior is non-trivial; not required for pure UI tweaks
- **Docs**: update `README.md` when adding user-visible features

## Files to Avoid Modifying Unnecessarily

- `poetry.lock` — only change via `poetry install` / `poetry update`
- `logs/data_cleaner.py` — generated output, not source of truth
- `.env` — local secrets only

## Common Tasks

| Task | Where to work |
|------|---------------|
| New summary column or stat | `get_input_data_summary()` |
| Change when a toggle is enabled | `get_cleaning_options()` |
| Adjust “dataset too small” warnings | `get_cleaning_size_warnings()` + `MIN_ROWS_*` constants |
| Improve “Applied / Not applied” detection | `_detect_*` functions + `_CLEANING_STEP_DETECTORS` |
| Change agent-generated code behavior | prompt in `create_data_cleaner_code()` |
| UI layout / new Streamlit section | `app.py` |
| User documentation | `README.md` |

## Verification

After changes, verify with:

```bash
poetry run streamlit run app.py
```

For utility logic:

```bash
poetry run python -c "
import pandas as pd
from data_cleaning_agent.utils import get_cleaning_options, get_cleaning_size_warnings
df = pd.read_csv('data/sample_data.csv')
print(get_cleaning_options(df))
print(get_cleaning_size_warnings(df, ['impute_missing_values', 'remove_high_missing_columns']))
"
```

Use `data/sample_data.csv` as the default test dataset.
