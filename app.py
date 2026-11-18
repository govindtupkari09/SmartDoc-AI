import os
from pathlib import Path

import faiss
import numpy as np
import pymupdf

from flask import Flask, jsonify, render_template, request
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "documents" / "aws_notes.pdf"

app = Flask(__name__)

embedding_model = None
faiss_index = None
document_chunks = []
chunk_metadata = []


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_pages_from_pdf(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pages = []

    with pymupdf.open(str(pdf_path)) as document:
        print(f"Total PDF pages: {len(document)}")

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text().strip()

            if page_text:
                pages.append({
                    "page": page_number,
                    "text": page_text
                })

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF."
        )

    print(f"Pages containing text: {len(pages)}")

    return pages


# ==========================================
# CREATE TEXT CHUNKS
# ==========================================

def create_chunks_from_pages(
    pages,
    chunk_size=1000,
    overlap=200
):
    chunks = []
    metadata = []

    step = chunk_size - overlap

    for page in pages:
        page_text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(page_text):
            chunk = page_text[
                start:start + chunk_size
            ].strip()

            if chunk:
                chunks.append(chunk)

                metadata.append({
                    "page": page_number
                })

            start += step

    print(f"Total chunks created: {len(chunks)}")

    return chunks, metadata


# ==========================================
# INITIALIZE RAG SYSTEM
# ==========================================

def initialize_rag():
    global embedding_model
    global faiss_index
    global document_chunks
    global chunk_metadata

    print("\nInitializing SmartDoc AI...")

    pages = extract_pages_from_pdf(PDF_PATH)

    document_chunks, chunk_metadata = (
        create_chunks_from_pages(pages)
    )

    if not document_chunks:
        raise ValueError(
            "No document chunks were created."
        )

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = embedding_model.encode(
        document_chunks,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    faiss_index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    faiss_index.add(embeddings)

    print(
        f"Vectors stored in FAISS: {faiss_index.ntotal}"
    )

    print("SmartDoc AI is ready!")


# ==========================================
# RETRIEVE RELEVANT DOCUMENT CONTEXT
# ==========================================

def retrieve_context(question, top_k=3):
    if embedding_model is None or faiss_index is None:
        raise RuntimeError(
            "RAG system has not been initialized."
        )

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    k = min(top_k, faiss_index.ntotal)

    distances, indices = faiss_index.search(
        question_embedding,
        k
    )

    results = []

    for chunk_index in indices[0]:
        if 0 <= chunk_index < len(document_chunks):
            results.append({
                "text": document_chunks[chunk_index],
                "page": chunk_metadata[chunk_index]["page"]
            })

    return results


# ==========================================
# GENERATE ANSWER USING GEMINI
# ==========================================

def ask_gemini(question, context_results):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable "
            "is not configured."
        )

    if not context_results:
        return (
            "I could not find that information "
            "in the document."
        )

    context_parts = []

    for source in context_results:
        context_parts.append(
            f"PDF Page {source['page']}:\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are SmartDoc AI, a helpful PDF document assistant.

Answer the user's question using only the supplied
document context.

Instructions:
1. Give a clear and useful answer.
2. Do not invent facts.
3. Do not invent page numbers.
4. If the context does not contain the answer, say:
   "I could not find that information in the document."
5. Do not claim that you searched the entire PDF when
   only selected context chunks were supplied.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = response.text

    if not answer:
        return (
            "The AI did not return an answer. "
            "Please try again."
        )

    return answer


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================
# CHAT API
# ==========================================

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON request is required."
        }), 400

    question = data.get("question", "")

    if not isinstance(question, str):
        return jsonify({
            "error": "Question must be text."
        }), 400

    question = question.strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if len(question) > 2000:
        return jsonify({
            "error": (
                "Question is too long. "
                "Maximum length is 2000 characters."
            )
        }), 400

    try:
        context_results = retrieve_context(question)

        answer = ask_gemini(
            question,
            context_results
        )

        sources = [
            {
                "text": source["text"],
                "page": source["page"]
            }
            for source in context_results
        ]

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception:
        app.logger.exception(
            "Error while processing the question"
        )

        return jsonify({
            "error": (
                "Unable to process your question. "
                "Check the VS Code terminal for details."
            )
        }), 500


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":
    initialize_rag()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )

