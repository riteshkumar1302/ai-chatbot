from rank_bm25 import BM25Okapi
import streamlit as st
from pypdf import PdfReader
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Onboarding Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hello Persistonaut")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 900px;
        padding-top: 16vh;
        margin-left: auto;
        margin-right: auto;
    }

    h1 {
        text-align: center;
    }

    .kiwi-label {
        font-family: Arial, Helvetica, sans-serif;
        color: #f5f5f5;
        font-size: 21px;
        font-weight: 700;
        line-height: 1.2;
        margin: 0 auto 14px auto;
        max-width: 1000px;
        text-align: center;
    }

    .kiwi-brand {
        color: #ff4b4b;
        font-family: Impact, "Arial Black", "Trebuchet MS", sans-serif;
        font-size: 24px;
        font-weight: 900;
        letter-spacing: 0;
    }

    [data-testid="stTextInput"] {
        width: min(760px, 92vw);
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stTextInput"] > div {
        width: 100%;
    }

    [data-testid="stTextInput"] input {
        text-align: center;
    }

    [data-testid="stTextInput"] label {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

pdf_folder = "pdfs"

# -------------------------------
# CONVERSATION MEMORY
# -------------------------------
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

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
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


@st.cache_resource
def load_reranker():

    return CrossEncoder(reranker_model_name)


def rerank_chunks(question, candidate_chunks, top_k=3):

    if not candidate_chunks:
        return []

    reranker = load_reranker()

    pairs = [
        (question, chunk)
        for chunk in candidate_chunks
    ]

    scores = reranker.predict(pairs)

    ranked_chunks = sorted(
        zip(candidate_chunks, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    return ranked_chunks[:top_k]


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

# -------------------------------
# BM25 INDEX
# -------------------------------
tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

llm = load_llm()

# -------------------------------
# CLEAN CONTEXT FOR LLM
# -------------------------------
def clean_context_for_llm(context):

    remove_patterns = [
        "QUICK REFERENCE",
        "RELATED TOPICS",
        "END OF RELATED TOPICS",
        "To find information about",
        "See Section"
    ]

    cleaned_lines = []

    for line in context.split("\n"):

        skip = False

        for pattern in remove_patterns:

            if pattern.lower() in line.lower():
                skip = True
                break

        if not skip:
            cleaned_lines.append(line)

    cleaned_context = "\n".join(cleaned_lines)

    return cleaned_context


def clean_answer_for_user(answer):

    answer = re.sub(
        r"\bconfidential\b",
        "restricted",
        answer,
        flags=re.IGNORECASE
    )

    remove_patterns = [
        "QUICK REFERENCE",
        "RELATED TOPICS",
        "END OF RELATED TOPICS",
        "To find information about",
        "See Section",
        "refer to section",
        "more details"
    ]

    cleaned_lines = []

    for line in answer.split("\n"):

        if any(pattern.lower() in line.lower() for pattern in remove_patterns):
            continue

        cleaned_lines.append(line)

    cleaned_answer = "\n".join(cleaned_lines).strip()

    return cleaned_answer or "Please reach out to your HR for this query"

# -------------------------------
# USER INPUT
# -------------------------------
st.markdown(
    '<div class="kiwi-label">I\'m <span class="kiwi-brand">KIWI-8</span>, How can I help you today</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "I'm KIWI-8, How can I help you today?",
    placeholder="Ask your HR related questions "
    ,
    label_visibility="collapsed"
)


# -------------------------------
# QUESTION ANSWERING
# -------------------------------
if question:

    original_question = question

    # ========== HANDLE FOLLOW-UP QUERIES ==========
    follow_up_phrases = ["give me", "whole", "complete", "correct", "full", "detailed", "in detail", "tell me more", "elaborate"]
    
    is_follow_up = any(phrase in question.lower() for phrase in follow_up_phrases)
    
    if is_follow_up and st.session_state.last_question:
        # User wants more details about previous question
        original_question = st.session_state.last_question
        question = original_question
        st.info(f"📌 Getting complete details about: {original_question}")
    # ==============================================

    # -------------------------------
    # STOP WORDS
    # -------------------------------
    stop_words = {
        "how", "to", "what", "is", "the",
        "a", "an", "please", "can", "i",
        "show", "tell", "me", "about",
        "in", "for", "of", "Please", "Can", "I", "Show", "Tell", "Me", "About",
        "In", "For", "Of", "help", 
    }

    # -------------------------------
    # CLEAN QUESTION
    # -------------------------------
    ignore_phrases = [
        "tell me in detail",
        "explain in detail",
        "give me details",
        "tell me more",
        "please explain",
        "please provide details",
        "in detail"
    ]

    question = question.lower()
    for phrase in ignore_phrases:
        question = question.replace(phrase, "")

    question = (
        question.replace("?", "")
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

    # ========== QUERY EXPANSION ==========
    intent_map = {
        "quit": ["resign", "resignation", "separation", "notice period", "full and final"],
        "masters": ["mtech", "m.tech", "higher education", "post graduate"],
        "mba": ["master of business administration"],
        "bonus": ["annual performance bonus", "apb", "incentive"],
        "insurance": ["mediclaim", "health insurance", "hospitalization"],
        "refer": ["referral", "employee referral", "referral bonus"],
    }
    
    original_lower = original_question.lower()
    for intent, keywords in intent_map.items():
        if intent in original_lower:
            clean_question += " " + " ".join(keywords)
            break
    # ========== END OF QUERY EXPANSION ==========

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

        # BOOST FOR RELEVANT SECTIONS
        if any(term in chunk_lower for term in ["resignation", "notice period", "separation", "full and final", "mtech", "mba", "higher education"]):
            match_score += 5

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
    # COLLECT RETRIEVAL CANDIDATES
    # -------------------------------
    candidate_chunks = []
    seen_chunks = set()

    def add_candidate(chunk):

        if chunk not in seen_chunks:
            candidate_chunks.append(chunk)
            seen_chunks.add(chunk)

    for chunk, score in matched_chunks[:5]:
        add_candidate(chunk)

    query_tokens = clean_question.split()
    bm25_scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:5]

    for index in ranked_indices:

        if bm25_scores[index] > 0:
            add_candidate(chunks[index])

    docs = vector_store.similarity_search(
        clean_question,
        k=5
    )

    for doc in docs:
        add_candidate(doc.page_content)

    # -------------------------------
    # RERANK CANDIDATES BY FULL QUESTION
    # -------------------------------
    reranked_chunks = rerank_chunks(
        original_question,
        candidate_chunks,
        top_k=2
    )

    context = "\n\n".join(
        [chunk for chunk, score in reranked_chunks]
    )

    # -------------------------------
    # LIMIT CONTEXT
    # -------------------------------
    context = context[:2000]

    # REMOVE NAVIGATION REFERENCES
    context = clean_context_for_llm(context)

    # -------------------------------
    # NO MATCH FOUND
    # -------------------------------
    if not context.strip():

        st.subheader("AI Answer")

        st.write(
            "Please reach out to your HR for this query"
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
- Provide relevant answer with key points (use bullet points for multiple items).
- Keep answers clean, readable, and properly formatted.
- Put every list item on a NEW LINE.
- Never merge multiple items into one paragraph.
- Use bullet points whenever possible.
- Preserve section headings.
- Ignore QUICK REFERENCE sections.
- Ignore RELATED TOPICS sections.
- Ignore "See Section" references.
- Do NOT include navigation references in final answer.
- Give direct answer only.
- Do not use the word "confidential"; use "restricted" or "internal" instead.
- Do NOT make up information.
- If answer is not found, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{original_question}

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
    st.subheader("Here is your answer")

    st.write(clean_answer_for_user(response.content))
    
    # ========== SAVE TO MEMORY ==========
    st.session_state.last_question = original_question
    st.session_state.last_answer = response.content
    # ===================================