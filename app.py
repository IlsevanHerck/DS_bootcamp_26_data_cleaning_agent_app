"""Streamlit interface for the Data Cleaning Agent."""

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from data_cleaning_agent import LightweightDataCleaningAgent
from data_cleaning_agent.utils import (
    build_cleaning_instructions,
    get_applied_cleaning_summary,
    get_cleaning_options,
    get_input_data_summary,
)

load_dotenv()

st.title("🧹 Data Cleaning Agent")

# Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Input Data Summary")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(get_input_data_summary(df_raw), use_container_width=True)

    st.subheader("Cleaning Options")
    st.caption(
        "Toggle on the steps you want the agent to consider. "
        "Options that do not apply to this dataset are disabled."
    )

    cleaning_options = get_cleaning_options(df_raw)
    selected_option_ids = []

    for option in cleaning_options:
        enabled = st.toggle(
            option["label"],
            value=option["default"] if option["applicable"] else False,
            disabled=not option["applicable"],
            help=option["description"] if option["applicable"] else option["reason"],
            key=f"cleaning_option_{option['id']}",
        )
        if option["applicable"]:
            if enabled:
                selected_option_ids.append(option["id"])
        else:
            st.caption(f"Not applicable: {option['reason']}")

    user_instructions = build_cleaning_instructions(selected_option_ids)

    # Clean button
    if st.button("Clean Data"):
        with st.spinner("Cleaning..."):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = LightweightDataCleaningAgent(model=llm, log=True)
            agent.invoke_agent(data_raw=df_raw, user_instructions=user_instructions)
            df_cleaned = agent.get_data_cleaned()
            
            st.success("Done!")

            st.subheader("Applied Cleaning Steps")
            st.dataframe(
                get_applied_cleaning_summary(df_raw, df_cleaned, selected_option_ids),
                use_container_width=True,
            )

            st.subheader("Cleaned Data Summary")
            st.write(f"Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            st.dataframe(get_input_data_summary(df_cleaned), use_container_width=True)

            st.subheader("Cleaned Data Preview")
            st.dataframe(df_cleaned.head(), use_container_width=True)
            
            # Download
            csv = df_cleaned.to_csv(index=False)
            st.download_button(
                "Download Cleaned Data",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
