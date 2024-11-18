import streamlit as st
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os


load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACESS_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token is missing. Please add it to the .env file.")

@st.cache_resource
def load_model():
    model_name = "gpt2" 
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    return model, tokenizer

def query_model(prompt, model, tokenizer, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.set_page_config(page_title="AI Data Analyzer", layout="wide")


st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #0000FF, #C6A7FF); /* Blue to Light Violet Gradient */
        padding: 20px;
        border-radius: 10px;
    }
    .main {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
    }
    .stButton button {
        background-color: #ff7f50;
        color: white;
        border-radius: 5px;
        border: none;
        font-size: 16px;
        padding: 8px 15px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #ff6347;
    }
    /* Floating Chatbot Icon on the bottom-right */
    .floating-chatbot {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #ff7f50;
        border-radius: 50%;
        padding: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        color: white;
        cursor: pointer;
        font-size: 20px;
        z-index: 1000;
    }
    /* Floating Robot Icon on the bottom-left */
    .floating-robot {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background-color: #4CAF50;
        border-radius: 50%;
        padding: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
        color: white;
        cursor: pointer;
        font-size: 20px;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("<h1 style='text-align: center;'>🌟 AI Data Analyzer 🌟</h1>", unsafe_allow_html=True)


st.sidebar.header("📜 Instructions")
st.sidebar.write("""
1. Upload a CSV or Excel (.xlsx) file.
2. Ensure your file has proper column headers.
3. Select the column to analyze and define your query.
4. Get results powered by GPT-2.
""")

uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
 
    if uploaded_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)


    if df.columns[0].startswith("Unnamed"):
        st.warning("⚠️ Detected unnamed columns. Renaming them for consistency.")
        num_columns = len(df.columns)
        df.columns = [f"Column {i}" for i in range(num_columns)]

    st.markdown("### 📋 Preview of Uploaded Data")
    st.dataframe(df.head())


    column_to_analyze = st.selectbox("📝 Select a column to analyze", df.columns)
    
    search_query = st.text_input("🔎 Enter your search query", placeholder="e.g., Provide details about ...")

    if st.button("🚀 Run Analysis"):
        st.write("⏳ Processing... This might take a while.")
        model, tokenizer = load_model()
        
        results = []
        for index, value in enumerate(df[column_to_analyze]):
            prompt = f"{search_query}: {value}"
            response = query_model(prompt, model, tokenizer)
            results.append({"Input": value, "Response": response})
        
    
        result_df = pd.DataFrame(results)
        st.markdown("### 🔍 Results")
        st.dataframe(result_df)
        
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="analysis_results.csv",
            mime="text/csv"
        )

st.markdown("<p style='text-align: center;'>💡 Powered by GPT-2 and Streamlit</p>", unsafe_allow_html=True)

st.markdown("<div class='floating-chatbot'>🤖</div>", unsafe_allow_html=True)

st.markdown("<div class='floating-robot'>🤖</div>", unsafe_allow_html=True)
