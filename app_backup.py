import os
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from google import genai


# =====================================================
# PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from all PDF pages.

    Returns:
        str: Combined text from PDF
    """

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found:\n{pdf_path}"
        )

    all_text = ""

    try:

        # IMPORTANT:
        # Keep document open while looping over pages
        with fitz.open(pdf_path) as doc:

            total_pages = len(doc)

            print(f"\nTotal Pages: {total_pages}")

            for page_number in range(total_pages):
                page = doc.load_page(page_number)
                page_text = page.get_text()

                if page_text.strip():
                    all_text += page_text + "\n"

    except Exception as error:
        raise RuntimeError(
            f"Error reading PDF: {error}"
        )

    if not all_text.strip():
        raise ValueError(
            "No extractable text found in PDF."
        )

    print(f"Total Characters Extracted: {len(all_text):,}")

    return all_text


# =====================================================
# TEXT CHUNKING
# =====================================================

def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
):
    """
    Split text into overlapping chunks.
    """

    if not text.strip():
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += (chunk_size - overlap)

    print(f"\nTotal Chunks Created: {len(chunks)}")

    return chunks


# =====================================================
# EMBEDDING MODEL
# =====================================================

def generate_embeddings(chunks):
    """
    Convert chunks into embeddings.
    """

    if not chunks:
        raise ValueError(
            "Cannot generate embeddings from empty chunks."
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Generating embeddings...")

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    print(
        f"Embedding Shape: {embeddings.shape}"
    )

    return embeddings, model


# =====================================================
# FAISS INDEX
# =====================================================

def build_faiss_index(embeddings):
    """
    Build vector index.
    """

    if embeddings.size == 0:
        raise ValueError(
            "Embeddings are empty."
        )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    print(
        f"\nVectors Stored In FAISS: {index.ntotal}"
    )

    return index


# =====================================================
# SEARCH
# =====================================================

def retrieve_context(
    question,
    model,
    index,
    chunks,
    top_k=3
):
    """
    Retrieve top chunks from FAISS.
    """

    if index.ntotal == 0:
        raise ValueError(
            "FAISS index is empty."
        )

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding,
        dtype=np.float32
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    print("\nRetrieved Chunks:")
    print("-" * 60)

    for idx in indices[0]:

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        chunk = chunks[idx]

        retrieved_chunks.append(chunk)

        print(chunk[:300])
        print("\n" + "-" * 60)

    return retrieved_chunks


# =====================================================
# GEMINI ANSWERING
# =====================================================

def ask_gemini(
    question,
    context_chunks
):
    """
    Query Gemini using retrieved context.
    """

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not found."
        )

    context = "\n\n".join(
        context_chunks
    )

    prompt = f"""
You are a helpful document assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
say:

"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as error:

        raise RuntimeError(
            f"Gemini API Error: {error}"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    pdf_path = "documents/aws_notes.pdf"

    print("=" * 60)
    print("SMARTDOC AI - PDF QUESTION ANSWERING")
    print("=" * 60)

    try:

        # 1. Extract PDF text
        text = extract_text_from_pdf(
            pdf_path
        )

        # 2. Chunking
        chunks = create_chunks(
            text,
            chunk_size=1000,
            overlap=200
        )

        # 3. Embeddings
        embeddings, embedding_model = generate_embeddings(
            chunks
        )

        # 4. Build FAISS
        index = build_faiss_index(
            embeddings
        )

        print(
            "\nSystem Ready."
        )

        while True:

            print("\n")
            question = input(
                "Ask a question (or type exit): "
            ).strip()

            if question.lower() == "exit":
                print("Goodbye.")
                break

            if not question:
                continue

            # 5. Retrieve
            retrieved_chunks = retrieve_context(
                question,
                embedding_model,
                index,
                chunks,
                top_k=3
            )

            # 6. Gemini QA
            answer = ask_gemini(
                question,
                retrieved_chunks
            )

            print("\nAnswer:")
            print("=" * 60)
            print(answer)
            print("=" * 60)

    except Exception as error:
        print("\nERROR:")
        print(error)


if __name__ == "__main__":
    main()