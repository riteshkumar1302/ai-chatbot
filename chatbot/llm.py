from __future__ import annotations

import re

from langchain_groq import ChatGroq


def load_llm(groq_api_key: str, model_name: str) -> ChatGroq:
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )


def build_prompt(context: str, question: str) -> str:
    return f"""
You are KIWI-8, an AI onboarding assistant.

Rules:
- Answer ONLY from the provided context.
- Understand similar wording and user intent.
- Prioritize headings and matching keywords.
- Provide relevant answers with key points.
- Use bullet points for multiple items.
- Put every list item on a new line.
- Preserve useful section headings.
- Ignore QUICK REFERENCE sections.
- Ignore RELATED TOPICS sections.
- Ignore "See Section" references.
- Do not include navigation references in the final answer.
- Give the direct answer only.
- Do not use the word "confidential"; use "restricted" or "internal" instead.
- Do not make up information.
- If the answer is not found, say: "I could not find that information in the documents."

Context:
{context}

Question:
{question}

Format guidance:
- Keep the answer clean and readable.
- Use short paragraphs only when bullets are not needed.
- For lists, keep each item on a separate line.
""".strip()


def clean_answer_for_user(answer: str) -> str:
    answer = re.sub(
        r"\bconfidential\b",
        "restricted",
        answer,
        flags=re.IGNORECASE,
    )

    remove_patterns = [
        "QUICK REFERENCE",
        "RELATED TOPICS",
        "END OF RELATED TOPICS",
        "To find information about",
        "See Section",
        "refer to section",
        "more details",
    ]

    cleaned_lines = []
    for line in answer.splitlines():
        if any(pattern.lower() in line.lower() for pattern in remove_patterns):
            continue
        cleaned_lines.append(line)

    cleaned_answer = "\n".join(cleaned_lines).strip()
    if "I could not find that information in the documents." in cleaned_answer:
        return "Please reach out to your HR for this query."

    return cleaned_answer or "Please reach out to your HR for this query."

