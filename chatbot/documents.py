from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdf_text(pdf_folder: str) -> str:
    folder = Path(pdf_folder)
    if not folder.exists():
        return ""

    all_text: list[str] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

    return "\n".join(all_text)


def split_text(all_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(all_text)

