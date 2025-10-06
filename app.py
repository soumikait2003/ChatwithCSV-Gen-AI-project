import streamlit as st
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import SmartDataframe

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()


def chat_with_csv(df, prompt):
    try:
        llm = OpenAI(api_token=openai_api_key)
        pandas_ai = SmartDataframe(df, config={"llm": llm})
        result = pandas_ai.chat(prompt)

        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, (list, dict)):
            try:
                return pd.DataFrame(result)
            except Exception as e:
                st.warning(f"Could not convert result to DataFrame: {e}")
                return result
        else:
            return result
    except Exception as e:
        st.error(f"Error during chat: {e}")
        return None


st.set_page_config(layout='wide')
st.title("ChatCSV powered by LLM")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is None:
    st.info("Please upload a CSV file to proceed.")
else:
    try:
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    input_text = st.text_area("Enter your query")
    if input_text and st.button("Chat with CSV"):
        st.info(f"Your Query: {input_text}")
        result = chat_with_csv(data, input_text)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif result:
            st.success(result)
