import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from exa_py import Exa
from langchain_together import Together

# === Load secrets ===
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
EXA_API_KEY = st.secrets["EXA_API_KEY"]

# === Init clients ===
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=TOGETHER_API_KEY,
    temperature=0.2,
    max_tokens=1024,
)

exa = Exa(api_key=EXA_API_KEY)

# === Streamlit UI ===
st.set_page_config(page_title="VC Analysis Agent")
st.title("📊 VC Analysis Agent")

uploaded_file = st.file_uploader("Upload a PDF pitch deck", type=["pdf"])

question = st.text_input("Ask a question about the company")

if uploaded_file and question:
    with st.spinner("🔎 Parsing PDF..."):
        # Save temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load and extract text
        loader = UnstructuredFileLoader("temp.pdf")
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        doc = Document(page_content=text)

    # === Try answering from PDF ===
    try:
        st.info("📄 Trying to answer from the PDF...")
        prompt = PromptTemplate.from_template("""
            You're a VC analyst. Use the following context to answer the question.

            Context:
            {context}

            Question:
            {question}

            Answer in 3-5 bullet points.
        """)
        chain: RunnableSequence = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": doc.page_content,
            "question": question
        })
        st.success("✅ Answered from PDF")
        st.markdown(response)

    except Exception as e:
        st.warning(f"[!] Failed to answer from PDF: {e}")

        # === Fallback: Exa Search ===
        try:
            st.info("🌐 Searching Exa...")
            results = exa.search(question).results[:3]
            urls = "\n".join([f"- [{r.title}]({r.url})" for r in results])

            prompt = PromptTemplate.from_template("""
                You're a VC analyst. Using the following links, answer the question.

                Links:
                {context}

                Question:
                {question}

                Be concise and give 3-5 bullet points.
            """)
            chain: RunnableSequence = prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": urls,
                "question": question
            })

            st.success("🌐 Answered using Exa search")
            st.markdown(response)

        except Exception as e:
            st.error(f"[❌] Exa search also failed: {e}")
