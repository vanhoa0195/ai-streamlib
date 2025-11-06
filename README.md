# Vietnam Airlines Chatbot (FastAPI + Streamlit)

This project provides a simple chatbot backend (FastAPI) and a Streamlit UI that uses Azure OpenAI (via the `openai` package) to answer questions about Vietnam Airlines. It uses the `.env` file for credentials.

Files added:
- `server.py` - FastAPI backend exposing POST /chat
- `streamlit_app.py` - Streamlit frontend to interact with the backend
- `requirements.txt` - Python dependencies

Environment:
1. Create a `.env` file in the project root with the following variables:
   - AZURE_OPENAI_ENDPOINT
   - AZURE_OPENAI_API_KEY

Run the backend (serves on port 3000):

```powershell
# Install deps
pip install -r requirements.txt

# Start server on port 3000
$env:AZURE_OPENAI_ENDPOINT = "https://..."
$env:AZURE_OPENAI_API_KEY = "sk-..."
uvicorn server:app --host 0.0.0.0 --port 3000
```

Run Streamlit UI:

```powershell
set BACKEND_URL=http://localhost:3000
streamlit run streamlit_app.py
```

API contract:
- POST /chat
  - Request body JSON: { userId: string, message: string, language?: "en" | "vi" }
  - Success response: 200 { reply: string }
  - Error responses: JSON { error: "..." } with appropriate HTTP status code

Notes and limitations:
- This implementation uses an in-memory session store (resets when the server restarts).
- The retriever for vietnamairlines.com is a naive HTML text extractor and may miss structured data; consider adding a proper scraper or cached dataset for production.
- The model is instructed to ground answers using a short excerpt from the site, but it may still hallucinate—verify critical information (prices, schedules) with the official site.
