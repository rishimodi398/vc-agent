# vc_agent.py

import os
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from exa_py import Exa
from together import Together

# === Load env vars ===
load_dotenv()
EXA_API_KEY = os.getenv("EXA_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# === Together LLM wrapper ===
class TogetherLLM:
    def __init__(self, model, api_key):
        self.client = Together(api_key=api_key)
        self.model = model

    def invoke(self, input, **kwargs):
        prompt = input.to_string() if hasattr(input, "to_string") else str(input)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content

# === Init clients ===
exa = Exa(EXA_API_KEY)
llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=TOGETHER_API_KEY)
llm_runnable = RunnableLambda(lambda x: llm.invoke(x))

# === PDF loader ===
def load_doc_text(file_path):
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    return docs

# === Answer from PDF ===
def try_answer_from_doc(query, docs):
    prompt = PromptTemplate.from_template("""
        You are a VC analyst. Use the following context to answer the question below.

        Context:
        {context}

        Question:
        {question}

        Provide 3-5 bullet points.
    """)
    doc_text = "\n".join([doc.page_content for doc in docs])
    chain = prompt | llm_runnable | StrOutputParser()
    return chain.invoke({"context": doc_text, "question": query})

# === Exa search ===
def answer_from_exa(query):
    prompt = PromptTemplate.from_template("""
        You are a VC analyst. Use external search results below to answer the user's question.

        Search results:
        {context}

        Question:
        {question}

        Provide 3-5 bullet points.
    """)
    results = exa.search(query).results[:3]
    links = "\n".join([f"{r.title}: {r.url}" for r in results])
    chain = prompt | llm_runnable | StrOutputParser()
    return chain.invoke({"context": links, "question": query})

# === Main ===
if __name__ == "__main__":
    file_path = input("📄 Drop your PDF file path: ").strip().strip('"')
    if not os.path.exists(file_path):
        print("❌ File not found.")
        exit()

    docs = load_doc_text(file_path)
    print("✅ File loaded.\n")

    while True:
        question = input("❓ Ask your question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break

        print("🔎 Searching in the PDF...")
        try:
            doc_answer = try_answer_from_doc(question, docs)
            print("\n📘 From PDF and Exa:\n", doc_answer)
        except Exception as e:
            print(f"[!] Failed to answer from PDF: {e}")
            print("\n🌐 Falling back to Exa search...")
            try:
                exa_answer = answer_from_exa(question)
                print("\n🌐 From Exa:\n", exa_answer)
            except Exception as e2:
                print(f"[!] Exa search also failed: {e2}")
