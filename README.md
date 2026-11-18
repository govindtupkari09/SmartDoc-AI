SmartDoc AI – Intelligent PDF Question Answering System
SmartDoc AI is an AI-powered document question-answering application that helps users understand information inside PDF files. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document content and generate context-aware answers with Google Gemini.
Key Features
- PDF Text Extraction: Extracts text from PDF documents using PyMuPDF.
- Document Chunking: Splits extracted text into smaller chunks for focused retrieval.
- Semantic Embeddings: Uses Sentence Transformers to convert text chunks into vector embeddings.
- Vector Search: Uses FAISS to index embeddings and retrieve relevant document sections.
- RAG-Based Answers: Provides retrieved document context to Google Gemini to generate relevant answers.
- PDF Question Answering: Lets users ask questions about the content of their PDF documents.
- Context-Grounded Responses: Uses retrieved document passages as context when generating answers.
Technology Stack
Area	Technologies
Programming Language	Python
PDF Processing	PyMuPDF
Embeddings	Sentence Transformers
Vector Search	FAISS
Generative AI	Google Gemini API
AI Architecture	Retrieval-Augmented Generation (RAG)


How It Works
1. The user provides a PDF document.
2. PyMuPDF extracts text from the PDF.
3. The extracted text is split into smaller chunks.
4. Sentence Transformers generates embeddings for the chunks.
5. FAISS indexes the embeddings for similarity search.
6. When the user asks a question, the system retrieves relevant chunks.
7. The retrieved context is provided to Google Gemini to generate an answer.
PDF Document
     |
     v
Text Extraction (PyMuPDF)
     |
     v
Text Chunking
     |
     v
Embeddings (Sentence Transformers)
     |
     v
FAISS Vector Index
     |
     v
User Question
     |
     v
Similarity Search / Context Retrieval
     |
     v
Google Gemini
     |
     v
Context-Aware Answer
Project Setup
1. Clone the repository
git clone https://github.com/govindtupkari09/SmartDoc-AI.git
cd SmartDoc-AI
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
If the repository contains a requirements.txt file:
pip install -r requirements.txt
Otherwise, install the core packages used by the project:
pip install pymupdf sentence-transformers faiss-cpu google-genai python-dotenv
Install any additional packages required by the application.
4. Configure the Gemini API key
Create a .env file in the configuration location expected by the application and add:
GEMINI_API_KEY=your_gemini_api_key
Use your own API key. Do not commit .env files or expose API keys in source code.
5. Run the application
Start the backend using the entry-point and command configured in the repository. If the project has a separate frontend, start it using that frontend's instructions.
The exact start command depends on the entry-point and framework configured in the repository.

Example Use Case
A user uploads an AWS whitepaper PDF and asks a question about a topic covered in the document. SmartDoc AI retrieves relevant passages and uses that context to generate an answer.
Security Notes
- Keep the Gemini API key in environment variables.
- Add .env, virtual environments, and generated index files to .gitignore where appropriate.
- Do not upload confidential documents or sensitive information unless you are authorized to do so.
Project Status
SmartDoc AI is an AI-powered PDF question-answering project built around document extraction, semantic retrieval, vector search, and Gemini-based response generation.
Author
Govind Tupkari
GitHub: govindtupkari09
If you find this project useful, consider giving the repository a star.
