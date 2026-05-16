import streamlit as st

from chatbot.config import get_groq_api_key, load_settings
from chatbot.documents import load_pdf_text, split_text
from chatbot.llm import build_prompt, clean_answer_for_user, load_llm
from chatbot.retrieval import HybridRetriever


st.set_page_config(
    page_title="AI Onboarding Chatbot",
    layout="wide",
)

st.title("Hello Persistonaut")

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
    unsafe_allow_html=True,
)


settings = load_settings()


@st.cache_data(show_spinner="Reading PDF documents...")
def cached_pdf_text(pdf_folder: str) -> str:
    return load_pdf_text(pdf_folder)


@st.cache_data(show_spinner="Preparing document chunks...")
def cached_chunks(all_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return split_text(all_text, chunk_size, chunk_overlap)


@st.cache_resource(show_spinner="Building retrieval index...")
def cached_retriever(chunks_tuple: tuple[str, ...], settings_snapshot) -> HybridRetriever:
    return HybridRetriever.from_chunks(list(chunks_tuple), settings_snapshot)


@st.cache_resource(show_spinner="Connecting to language model...")
def cached_llm(groq_api_key: str, model_name: str):
    return load_llm(groq_api_key, model_name)


def resolve_follow_up_question(question: str) -> str:
    follow_up_phrases = {
        "complete",
        "correct",
        "detailed",
        "elaborate",
        "full",
        "give me",
        "in detail",
        "tell me more",
        "whole",
    }

    if (
        any(phrase in question.lower() for phrase in follow_up_phrases)
        and st.session_state.last_question
    ):
        st.info(f"Getting complete details about: {st.session_state.last_question}")
        return st.session_state.last_question

    return question


if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


all_text = cached_pdf_text(settings.pdf_folder)
if not all_text:
    st.error(f"No PDF content found in '{settings.pdf_folder}'.")
    st.stop()

chunks = cached_chunks(all_text, settings.chunk_size, settings.chunk_overlap)
if not chunks:
    st.error("PDF content was found, but no searchable text chunks could be created.")
    st.stop()

groq_api_key = get_groq_api_key(st.secrets)
if not groq_api_key:
    st.error("Groq API key not found. Set GROQ_API_KEY as an environment variable.")
    st.stop()

retriever = cached_retriever(tuple(chunks), settings)
llm = cached_llm(groq_api_key, settings.groq_model_name)

st.markdown(
    '<div class="kiwi-label">I\'m <span class="kiwi-brand">KIWI-8</span>, How can I help you today?</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "I'm KIWI-8, How can I help you today?",
    placeholder="Ask your HR related questions",
    label_visibility="collapsed",
)

if question:
    original_question = resolve_follow_up_question(question)
    context = retriever.retrieve_context(original_question)

    st.subheader("Here is your answer")

    if not context.strip():
        st.write("Please reach out to your HR for this query.")
        st.stop()

    prompt = build_prompt(context, original_question)
    with st.spinner("Thinking..."):
        response = llm.invoke(prompt)

    answer = clean_answer_for_user(response.content)
    st.write(answer)

    st.session_state.last_question = original_question
    st.session_state.last_answer = answer
