# Data Cleaning Agent

An AI-powered data cleaning agent that automatically cleans messy datasets using LangChain and LangGraph. The agent uses an LLM to generate and execute Python code for common data cleaning tasks like handling missing values, removing duplicates, and dropping low-quality columns.

## How It Works

The agent follows a simple workflow:
1. **Analyze**: Examines your dataset structure and identifies data quality issues
2. **Generate**: Uses an LLM to create custom Python cleaning code based on the data
3. **Execute**: Runs the generated code to clean your data
4. **Retry**: Automatically fixes errors if the generated code fails (up to 3 attempts)

This approach combines the flexibility of LLMs with the reliability of pandas operations.

## Setup

### Prerequisites

- **Python 3.9 or higher** (3.9, 3.10, 3.11, 3.12, or 3.13) - **Note**: Python 3.9.7 is not supported due to a Streamlit compatibility issue
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.9, python3.10, python3.12, etc.

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Streamlit Web Interface

The easiest way to use the agent is through the web interface:

```bash
poetry run streamlit run app.py
```

Then:
1. Upload your CSV file
2. Review the **Input Data Summary** — a per-column overview of datatype, average, min, max, and % filled
3. Choose **Cleaning Options** — toggle the steps you want the agent to apply (inapplicable steps are disabled automatically)
4. Click "Clean Data"
5. Review the **Applied Cleaning Steps** — which selected steps took effect and what changed
6. Review the **Cleaned Data Summary** — the same column-level overview for the cleaned dataset
7. Download the cleaned dataset

#### Cleaning Options

After upload, the app inspects your dataset and shows common cleaning steps as toggles:

| Option | Enabled when… |
|--------|----------------|
| Remove columns with high missing values | Any column has >40% missing |
| Impute missing values | Missing values are present |
| Remove duplicate rows | Duplicate rows are found |
| Standardize text formatting | Text columns exist |
| Convert data types | Object columns look numeric or datetime-like |
| Remove outliers | Numeric columns exist (off by default) |

Applicable options are on by default (except outlier removal) and can be turned off. Options that do not apply to your data are disabled with a short explanation. Your selections are passed to the agent as cleaning instructions.

#### After Cleaning

Once cleaning completes, the app shows:

**Applied Cleaning Steps** — For each selected option, a table with:

| Field | Description |
|-------|-------------|
| **Step** | Cleaning step name |
| **Status** | `Applied` or `Not applied` |
| **Details** | What changed (e.g. columns removed, missing values filled, rows dropped) |

The app compares the raw and cleaned datasets to infer which steps had a measurable effect.

**Cleaned Data Summary** — The same structure as the input summary (column, datatype, average, min, max, % filled), so you can compare data quality before and after cleaning. A preview of the first rows is shown below the summary.

### Python API

For programmatic use or integration into data pipelines:

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent

# Initialize the agent with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = LightweightDataCleaningAgent(model=llm)

# Load your messy data
df = pd.read_csv("your_data.csv")

# Run the cleaning agent
agent.invoke_agent(data_raw=df)

# Get the cleaned dataset
cleaned_df = agent.get_data_cleaned()

# Save or use the cleaned data
cleaned_df.to_csv("cleaned_data.csv", index=False)
```

**Optional: Provide custom instructions**

```python
# Give specific cleaning instructions to the agent
agent.invoke_agent(
    data_raw=df,
    user_instructions="Remove columns with more than 30% missing values and standardize date formats"
)
```

Or build instructions from the same cleaning options used in the Streamlit app:

```python
from data_cleaning_agent.utils import get_cleaning_options, build_cleaning_instructions

options = get_cleaning_options(df)
selected = [
    option["id"]
    for option in options
    if option["applicable"] and option["default"]
]
user_instructions = build_cleaning_instructions(selected)

agent.invoke_agent(data_raw=df, user_instructions=user_instructions)

# Review what changed
from data_cleaning_agent.utils import get_applied_cleaning_summary, get_input_data_summary

applied = get_applied_cleaning_summary(df, cleaned_df, selected)
print(applied)
print(get_input_data_summary(cleaned_df))
```

### Utility Functions

#### `get_input_data_summary`

Build a per-column summary of a DataFrame. Used for both the input and cleaned data summaries in the Streamlit app. Returns a table with:

| Field | Description |
|-------|-------------|
| **Column** | Column name |
| **Datatype** | pandas dtype |
| **Average** | Mean value (numeric columns only) |
| **Minimum** | Minimum value (numeric and datetime columns) |
| **Maximum** | Maximum value (numeric and datetime columns) |
| **% Filled** | Percentage of rows with non-null values |

```python
import pandas as pd
from data_cleaning_agent.utils import get_input_data_summary

df = pd.read_csv("your_data.csv")
summary = get_input_data_summary(df)
print(summary)
```

This function is used by the Streamlit app after upload and after cleaning. You can also call it directly in scripts or notebooks to inspect data quality.

#### `get_cleaning_options`

Inspect a DataFrame and return which common cleaning steps apply. Each option includes:

| Field | Description |
|-------|-------------|
| **id** | Stable identifier (e.g. `remove_duplicates`) |
| **label** | Display name shown in the UI |
| **description** | What the step does |
| **applicable** | Whether the step makes sense for this dataset |
| **default** | Suggested toggle state when applicable |
| **reason** | Why the step is disabled when not applicable |

```python
from data_cleaning_agent.utils import get_cleaning_options

options = get_cleaning_options(df)
for option in options:
    status = "on" if option["applicable"] else "disabled"
    print(option["label"], status)
```

#### `build_cleaning_instructions`

Convert a list of selected option ids into agent instructions:

```python
from data_cleaning_agent.utils import build_cleaning_instructions

instructions = build_cleaning_instructions(["remove_duplicates", "impute_missing_values"])
agent.invoke_agent(data_raw=df, user_instructions=instructions)
```

If no options are selected, the agent is instructed to return the data unchanged.

#### `get_applied_cleaning_summary`

Compare raw and cleaned DataFrames to summarize which selected cleaning steps appear to have been applied:

| Field | Description |
|-------|-------------|
| **Step** | Cleaning step name |
| **Status** | `Applied` or `Not applied` |
| **Details** | Detected changes (columns dropped, values imputed, rows removed, etc.) |

```python
from data_cleaning_agent.utils import get_applied_cleaning_summary

summary = get_applied_cleaning_summary(
    df_raw=df,
    df_cleaned=cleaned_df,
    selected_option_ids=["remove_duplicates", "impute_missing_values"],
)
print(summary)
```

Detection is heuristic: it looks for measurable differences between the datasets (e.g. fewer missing values, dropped columns, changed dtypes). Use this alongside the cleaned data summary to validate the agent's output.

## Project Structure

```
data-cleaning-agent/
├── data_cleaning_agent/
│   ├── __init__.py
│   ├── data_cleaning_agent.py  # Main agent class
│   └── utils.py                # Utility functions (summaries, cleaning options, step detection)
├── app.py                      # Streamlit interface
├── pyproject.toml              # Dependencies configuration
├── poetry.lock                 # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.
