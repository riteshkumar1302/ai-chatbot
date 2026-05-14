import streamlit as st
from pypdf import PdfReader
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# PAGE CONFIG
st.set_page_config(
    page_title="AI Onboarding Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Onboarding Chatbot")

pdf_folder = "pdfs"

# -------------------------------
# LOAD PDF TEXT
# -------------------------------
@st.cache_data
def load_pdf_text(pdf_folder):

    all_text = ""

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder, file)

            reader = PdfReader(pdf_path)

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    all_text += text + "\n"

    return all_text


# -------------------------------
# SPLIT TEXT INTO CHUNKS
# -------------------------------
@st.cache_data
def split_text(all_text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_text(all_text)

    return chunks


# -------------------------------
# CREATE VECTOR STORE
# -------------------------------
faiss_index_path = "faiss_index"


@st.cache_resource
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=".cache/embeddings",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # LOAD EXISTING INDEX
    if os.path.exists(faiss_index_path):

        vector_store = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # CREATE NEW INDEX
    else:

        vector_store = FAISS.from_texts(chunks, embeddings)

        vector_store.save_local(faiss_index_path)

    return vector_store


# -------------------------------
# LOAD LLM
# -------------------------------
@st.cache_resource
def load_llm():

    # LOCAL PC
    groq_api_key = os.getenv("GROQ_API_KEY")

    # STREAMLIT CLOUD
    if not groq_api_key:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    return llm


# -------------------------------
# LOAD EVERYTHING
# -------------------------------
all_text = load_pdf_text(pdf_folder)

chunks = split_text(all_text)

vector_store = create_vector_store(chunks)

llm = load_llm()


# -------------------------------
# USER INPUT
# -------------------------------
question = st.text_input("Ask Your Question")


# -------------------------------
# QUESTION ANSWERING
# -------------------------------
if question:

    # LOWERCASE SEARCH
    question = question.lower().strip()

    # VECTOR SEARCH
    docs = vector_store.similarity_search(
        question,
        k=8
    )

    # CREATE CONTEXT
    context = "\n\n".join([doc.page_content for doc in docs])

    context = context[:5000]

    # PROMPT
    prompt = f"""
You are an AI onboarding assistant.

Rules:
- Answer ONLY from the provided context.
- If answer exists even partially, provide the answer.
- Ignore uppercase/lowercase differences.
- Understand similar wording.
- Keep answers clean and readable.
- Do NOT make up information.
- If answer is not found, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{question}
"""

    # LLM RESPONSE
    response = llm.invoke(prompt)

    # OUTPUT
    st.subheader("AI Answer")

    st.write(response.content)