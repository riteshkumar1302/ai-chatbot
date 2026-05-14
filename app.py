import streamlit as st
from pypdf import PdfReader
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Onboarding Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hello Persistonaut")

pdf_folder = "pdfs"

# -------------------------------
# LOAD PDF TEXT
# -------------------------------
@st.cache_data
def load_pdf_text(pdf_folder):

    all_text = ""

    if not os.path.exists(pdf_folder):
        return ""

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
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = text_splitter.split_text(all_text)

    return chunks


# -------------------------------
# VECTOR STORE
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

    # LOCAL SYSTEM
    groq_api_key = os.getenv("GROQ_API_KEY")

    # STREAMLIT CLOUD
    if not groq_api_key:
        groq_api_key = st.secrets["GROQ_API_KEY"]

    # CHECK API
    if not groq_api_key:

        st.error("Groq API key not found.")

        st.stop()

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

# HANDLE EMPTY PDF
if not all_text:

    st.error("No PDF content found.")

    st.stop()

chunks = split_text(all_text)

vector_store = create_vector_store(chunks)

llm = load_llm()


# -------------------------------
# USER INPUT
# -------------------------------
question = st.text_input(
    "I'm KIWI8, Ask Your HR Questions",
    placeholder="Ask you HR related questions about holidays, policies, benefits, onboarding, and more..."
)


# -------------------------------
# QUESTION ANSWERING
# -------------------------------
if question:

    # -------------------------------
    # STOP WORDS
    # -------------------------------
    stop_words = {
        "how", "to", "what", "is", "the",
        "a", "an", "please", "can", "i",
        "show", "tell", "me", "about",
        "in", "for", "of", "Please", "Can", "I", "Show", "Tell", "Me", "About",
        "In", "For", "Of", "help"
    }

    # -------------------------------
    # CLEAN QUESTION
    # -------------------------------
    question = (
        question.lower()
        .replace("?", "")
        .replace(",", "")
        .strip()
    )

    # -------------------------------
    # REMOVE STOP WORDS
    # -------------------------------
    words = question.split()

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    clean_question = " ".join(filtered_words)

    # -------------------------------
    # KEYWORD SEARCH
    # -------------------------------
    matched_chunks = []

    for chunk in chunks:

        chunk_lower = chunk.lower()

        match_score = 0

        # MATCH WORDS
        for word in filtered_words:

            if len(word) <= 2:
                continue

            if word in chunk_lower:
                match_score += 1

        # HEADING BOOST
        if clean_question in chunk_lower:
            match_score += 5

        # IMPORTANT SECTION BOOST
        important_keywords = [
            "holiday",
            "vpn",
            "password",
            "jira",
            "cafeteria",
            "menu",
            "onboarding"
        ]

        for keyword in important_keywords:

            if keyword in clean_question and keyword in chunk_lower:
                match_score += 2

        # STORE GOOD MATCHES
        if match_score >= 2:
            matched_chunks.append((chunk, match_score))

    # -------------------------------
    # SORT BEST MATCHES
    # -------------------------------
    matched_chunks = sorted(
        matched_chunks,
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------
    # CREATE CONTEXT
    # -------------------------------
    if matched_chunks:

        context = "\n\n".join(
            [chunk for chunk, score in matched_chunks[:3]]
        )

    # -------------------------------
    # FALLBACK TO FAISS SEARCH
    # -------------------------------
    else:

        docs = vector_store.similarity_search(
            clean_question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

    # -------------------------------
    # LIMIT CONTEXT
    # -------------------------------
    context = context[:5000]

    # -------------------------------
    # NO MATCH FOUND
    # -------------------------------
    if not context.strip():

        st.subheader("AI Answer")

        st.write(
            "I could not find that information in the documents."
        )

        st.stop()

    # -------------------------------
    # PROMPT
    # -------------------------------
    prompt = f"""
You are an AI onboarding assistant.

Rules:
- Answer ONLY from the provided context.
- Understand similar wording and user intent.
- Prioritize headings and matching keywords.
- Provide only the most relevant answer, limiting to 1-3 key points or items.
- Keep answers clean, readable, and properly formatted.
- Put every list item on a NEW LINE.
- Never merge multiple items into one paragraph.
- Use bullet points whenever possible.
- Preserve section headings.
- Do NOT make up information.
- If answer is not found, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{clean_question}

IMPORTANT RESPONSE FORMAT:

Example:

PUNE HOLIDAYS 2026

Total Holidays: 7

- New Year's Day - 1-Jan-26 - Thursday
- Pongal / Makar Sankranti - 14-Jan-26 - Wednesday
- Ram Navmi - 27-Mar-26 - Friday
- Good Friday - 3-Apr-26 - Friday

Always keep each item on a separate line.
"""

    # -------------------------------
    # LLM RESPONSE
    # -------------------------------
    with st.spinner("Thinking..."):

        response = llm.invoke(prompt)

    # -------------------------------
    # OUTPUT
    # -------------------------------
    st.subheader("AI Answer")

    st.write(response.content)