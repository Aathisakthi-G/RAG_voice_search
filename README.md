# RAG Audio Enabled (RAG with Voice Search)

## Overview

RAG Audio Enabled is a web-based Retrieval-Augmented Generation (RAG) application for biomedical document search and Q&A, featuring voice search. Users can upload medical documents (PDF, DOCX), perform semantic and keyword search, and interact with an LLM assistant using both text and voice queries. The app is designed for medical researchers and clinicians, with strict guardrails to avoid personal medical advice and illegal content.

---

## Features

- **Document Upload:** Upload PDF and DOCX files for semantic search and Q&A.
- **RAG Search:** Combines vector (semantic) and keyword search for accurate retrieval.
- **Voice Search:** Use your microphone to ask questions by voice (Chrome, Edge, Safari supported).
- **User Authentication:** Register and log in to manage your own documents and conversations.
- **Conversation History:** All Q&A sessions are saved and can be revisited.
- **Guardrails:** Ensures only medical, non-personal, and legal queries are answered.
- **Vector Management:** View and delete your uploaded document vectors.

---

## Tech Stack

- **Backend:** Flask, PyMongo (MongoDB), PyMilvus (Milvus vector DB), SentenceTransformers
- **Frontend:** HTML, CSS, JavaScript (with Web Speech API for voice search)
- **LLM:** OpenRouter API (Llama 3.3-70B-Instruct)

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd RAG_audio_enabled
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory with the following variables:

```
OPENROUTER_API_KEY=your_openrouter_api_key
MILVUS_HOST=localhost
MILVUS_PORT=19530
OPENAI_API_KEY=your_openai_api_key (optional, for fallback)
```

- **OPENROUTER_API_KEY**: Required for LLM responses (get from [OpenRouter](https://openrouter.ai/)).
- **MILVUS_HOST/PORT**: Milvus vector DB connection (default: localhost:19530).
- **OPENAI_API_KEY**: Optional, for fallback.

### 4. Start MongoDB and Milvus
- **MongoDB:** Ensure a MongoDB instance is running (default: `mongodb://3.110.121.81:27017/`).
- **Milvus:** Start Milvus (see [Milvus docs](https://milvus.io/docs/install_standalone-docker.md)).

### 5. Run the App
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. **Register/Login:** Create an account or log in.
2. **Upload Documents:** Use the sidebar to upload PDF/DOCX files.
3. **Ask Questions:**
   - Type your question or click the microphone button to use voice search.
   - The assistant will answer using only the uploaded documents.
4. **Manage Vectors:** Go to "Manage Vectors" to view/delete your document vectors.
5. **Conversation History:** All your Q&A sessions are saved for review.

### Voice Search
- Click the microphone button next to the input box.
- Speak your question; the recognized text will appear in the input.
- Submit to get an answer.
- Requires Chrome, Edge, or Safari and microphone permissions.

---

## File Structure

- `app.py` — Main Flask app
- `config/rails/` — Guardrails config, actions, and flows
- `templates/` — HTML templates (UI)
- `uploads/` — Uploaded files (if enabled)
- `requirements.txt` — Python dependencies

---

## Security & Guardrails
- Only answers questions based on uploaded documents.
- Refuses personal medical advice and illegal requests.
- User authentication required for all features.

---

## Troubleshooting
- **Speech recognition not working?** Use Chrome, Edge, or Safari and allow microphone access.
- **Milvus/MongoDB connection errors?** Ensure both services are running and accessible.
- **API key errors?** Check your `.env` file and ensure the keys are correct.

---

## License
MIT License 