# server.py
import os
import time
import logging
from typing import Dict, List, Any, Optional
import threading
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import json
import asyncio
import io
import numpy as np

# OpenAI/Azure client exceptions
from openai import AzureOpenAI, APIConnectionError, APIError, APITimeoutError, RateLimitError

# Optional TTS dependencies
try:
    from transformers import VitsModel, AutoTokenizer
    import torch
    TTS_DEPENDENCIES_AVAILABLE = True
except Exception:
    TTS_DEPENDENCIES_AVAILABLE = False

# LangChain optional imports (we'll handle import errors gracefully)
try:
    from langchain.chat_models import AzureChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import ChatPromptTemplate
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Load .env
load_dotenv()

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

DEFAULT_AZURE_OPENAI_ENDPOINT: str = ""
DEFAULT_AZURE_OPENAI_API_KEY: str = ""

_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or DEFAULT_AZURE_OPENAI_ENDPOINT
_azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or DEFAULT_AZURE_OPENAI_API_KEY

# Azure OpenAI client (original direct client)
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=_azure_endpoint,
    api_key=_azure_api_key,
)

DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "GPT-4.0-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))

# In-memory session store
SESSIONS: Dict[str, List[Dict[str, str]]] = {}
_HERE = os.path.dirname(__file__)
_CHATS_STORE = os.path.join(_HERE, "chats_store.json")
_SESSIONS_LOCK = threading.Lock()

def _load_sessions_from_disk() -> None:
    global SESSIONS
    try:
        if os.path.exists(_CHATS_STORE):
            with open(_CHATS_STORE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    for k, v in list(data.items()):
                        if not isinstance(v, list):
                            data.pop(k, None)
                    SESSIONS = data
    except Exception as e:
        logger.warning("Could not load chats store: %s", e)

def _save_sessions_to_disk() -> None:
    try:
        with _SESSIONS_LOCK:
            tmp_path = _CHATS_STORE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(SESSIONS, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, _CHATS_STORE)
    except Exception as e:
        logger.warning("Could not save chats store: %s", e)

_load_sessions_from_disk()

# vna_chromadb module expected to be provided by you (setup_vna_collection, query_vna_info)
from vna_chromadb import setup_vna_collection, query_vna_info

# Initialize ChromaDB collection
try:
    vna_collection = setup_vna_collection()
    logger.info("Successfully initialized Vietnam Airlines ChromaDB collection")
except Exception as e:
    logger.error("Failed to initialize ChromaDB collection: %s", e)
    vna_collection = None
    
import random
from datetime import datetime, timedelta

def generate_fake_vna_flights():
    """Generate realistic fake Vietnam Airlines flights for every day in the next 30 days."""
    known_routes = {
        ("Hanoi (HAN)", "Ho Chi Minh City (SGN)"): {"duration": 2.0, "base_price": 1800000},
        ("Hanoi (HAN)", "Da Nang (DAD)"): {"duration": 1.5, "base_price": 1500000},
        ("Hanoi (HAN)", "Bangkok (BKK)"): {"duration": 2.0, "base_price": 3500000},
        ("Hanoi (HAN)", "Singapore (SIN)"): {"duration": 3.5, "base_price": 5000000},
        ("Hanoi (HAN)", "Tokyo (NRT)"): {"duration": 5.0, "base_price": 8500000},
        ("Hanoi (HAN)", "Seoul (ICN)"): {"duration": 4.0, "base_price": 6500000},
        ("Hanoi (HAN)", "Paris (CDG)"): {"duration": 12.0, "base_price": 15000000},
        ("Ho Chi Minh City (SGN)", "Singapore (SIN)"): {"duration": 2.0, "base_price": 4000000},
        ("Ho Chi Minh City (SGN)", "Tokyo (NRT)"): {"duration": 5.5, "base_price": 9000000},
        ("Ho Chi Minh City (SGN)", "Sydney (SYD)"): {"duration": 8.5, "base_price": 12000000},
        ("Ho Chi Minh City (SGN)", "Melbourne (MEL)"): {"duration": 8.0, "base_price": 11000000},
        ("Ho Chi Minh City (SGN)", "Paris (CDG)"): {"duration": 13.0, "base_price": 15500000},
        ("Da Nang (DAD)", "Seoul (ICN)"): {"duration": 4.0, "base_price": 6000000},
        ("Da Nang (DAD)", "Singapore (SIN)"): {"duration": 2.5, "base_price": 4500000},
    }

    aircraft_types = ["Boeing 787-9 Dreamliner", "Airbus A350-900", "Airbus A321neo"]
    fare_classes = ["Economy", "Premium Economy", "Business"]

    flights = []
    today = datetime.now()

    # Generate flights for each route over the next 30 days
    for (origin, dest), data in known_routes.items():
        for day_offset in range(30):  # next 30 days
            dep_date = today + timedelta(days=day_offset)
            for _ in range(random.randint(1, 2)):  # 1–2 flights per day per route
                flight_no = f"VN{random.randint(100,999)}"
                dep_time = dep_date.replace(hour=random.randint(5, 22), minute=random.choice([0, 15, 30, 45]))
                duration_hours = data["duration"]
                arr_time = dep_time + timedelta(hours=duration_hours)
                aircraft = random.choice(aircraft_types)
                status = random.choice(["On Time", "Delayed", "Available", "Fully Booked"])
                weekly_freq = "Daily"

                base = data["base_price"]
                prices = {
                    "Economy": int(base + random.uniform(-0.1, 0.1) * base),
                    "Premium Economy": int(base * 1.4 + random.uniform(-0.1, 0.1) * base),
                    "Business": int(base * 2.0 + random.uniform(-0.1, 0.1) * base),
                }

                flights.append({
                    "flight_no": flight_no,
                    "origin": origin,
                    "destination": dest,
                    "departure": dep_time.strftime("%Y-%m-%d %H:%M"),
                    "arrival": arr_time.strftime("%Y-%m-%d %H:%M"),
                    "aircraft": aircraft,
                    "duration_hr": duration_hours,
                    "status": status,
                    "weekly_frequency": weekly_freq,
                    "prices": prices,
                })

    return flights


FAKE_FLIGHTS = generate_fake_vna_flights()


import re
import random
from datetime import datetime, timedelta

def extract_date_from_text(text: str):
    """Try to extract a date from the query (e.g. 'Nov 15', '2025-11-20', 'tomorrow', 'next week')."""
    text = text.lower()
    today = datetime.now()
    # Relative references
    if "tomorrow" in text:
        return today + timedelta(days=1)
    if "next week" in text:
        return today + timedelta(days=7)

    # Match formats like "Nov 15", "15 Nov", "2025-11-20"
    date_patterns = [
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
        r"([A-Za-z]+)\s+(\d{1,2})",
        r"(\d{1,2})\s+([A-Za-z]+)"
    ]
    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            try:
                s = " ".join(m.groups())
                return datetime.strptime(s, "%Y %m %d")
            except Exception:
                try:
                    return datetime.strptime(s, "%d %m %Y")
                except Exception:
                    # Try month name formats
                    try:
                        return datetime.strptime(s, "%b %d")
                    except Exception:
                        try:
                            return datetime.strptime(s, "%d %b")
                        except Exception:
                            continue
    return None


def query_fake_flights(query: str, limit: int = 5):
    """Smarter search for flights based on user query, with route + date filtering."""
    query_lower = query.lower()
    target_date = extract_date_from_text(query)
    matches = []

    # Detect potential origin/destination cities
    city_keywords = {
        "hanoi": "Hanoi (HAN)",
        "ho chi minh": "Ho Chi Minh City (SGN)",
        "saigon": "Ho Chi Minh City (SGN)",
        "danang": "Da Nang (DAD)",
        "da nang": "Da Nang (DAD)",
        "nha trang": "Nha Trang (CXR)",
        "phu quoc": "Phu Quoc (PQC)",
        "tokyo": "Tokyo (NRT)",
        "seoul": "Seoul (ICN)",
        "singapore": "Singapore (SIN)",
        "bangkok": "Bangkok (BKK)",
        "paris": "Paris (CDG)",
        "london": "London (LHR)",
        "sydney": "Sydney (SYD)",
        "melbourne": "Melbourne (MEL)",
        "new york": "New York (JFK)",
        "san francisco": "San Francisco (SFO)",
    }

    origin = None
    destination = None
    for city_kw, city_full in city_keywords.items():
        if city_kw in query_lower:
            if origin is None:
                origin = city_full
            elif destination is None and city_full != origin:
                destination = city_full

    # Search flights
    for f in FAKE_FLIGHTS:
        # Route filter
        if origin and origin not in f["origin"]:
            continue
        if destination and destination not in f["destination"]:
            continue

        # Date filter (±1 day window)
        if target_date:
            flight_date = datetime.strptime(f["departure"], "%Y-%m-%d %H:%M")
            if abs((flight_date - target_date).days) > 1:
                continue

        matches.append(f)

    # Fallback: if no matches, relax criteria
    if not matches:
        for f in FAKE_FLIGHTS:
            if any(
                kw in query_lower
                for kw in [
                    f["origin"].split()[0].lower(),
                    f["destination"].split()[0].lower(),
                    f["origin"].split("(")[1][:3].lower(),
                    f["destination"].split("(")[1][:3].lower(),
                ]
            ):
                matches.append(f)

    # Shuffle to avoid repetition
    random.shuffle(matches)
    return matches[:limit]


def format_flight_results(flights: list) -> str:
    """Return a readable text summary for the assistant."""
    if not flights:
        return "Sorry, I couldn’t find any flights matching your request."

    msg_lines = ["Here are some Vietnam Airlines flights you might be interested in:\n"]
    for f in flights:
        msg_lines.append(
            f"✈️ **{f['flight_no']}** — {f['origin']} → {f['destination']} "
            f"({f['aircraft']}, {f['duration_hr']}h)\n"
            f"🕓 Departure: {f['departure']} | Arrival: {f['arrival']} | Frequency: {f.get('weekly_frequency','Daily')}\n"
            f"💺 Status: {f['status']}\n"
            f"💰 Prices: Economy {f['prices']['Economy']:,}₫, "
            f"Premium {f['prices']['Premium Economy']:,}₫, Business {f['prices']['Business']:,}₫\n"
        )
    return "\n".join(msg_lines)

    """Return a readable text summary for the assistant."""
    if not flights:
        return "Sorry, I couldn’t find any flights matching your request."
    msg_lines = ["Here are some Vietnam Airlines flights you might be interested in:\n"]
    for f in flights:
        msg_lines.append(
            f"✈️ **{f['flight_no']}** — {f['origin']} → {f['destination']} "
            f"({f['aircraft']}, {f['duration_hr']}h)\n"
            f"🕓 Departure: {f['departure']} | Arrival: {f['arrival']}\n"
            f"💺 Status: {f['status']}\n"
            f"💰 Prices: Economy {f['prices']['Economy']:,}₫, "
            f"Premium {f['prices']['Premium Economy']:,}₫, Business {f['prices']['Business']:,}₫\n"
        )
    return "\n".join(msg_lines)


# TTS model load
SAMPLE_RATE = 22050
TTS_AVAILABLE = False
if TTS_DEPENDENCIES_AVAILABLE:
    try:
        tts_model = VitsModel.from_pretrained("facebook/mms-tts-vie")
        tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
        TTS_AVAILABLE = True
        logger.info("TTS model loaded successfully")
    except Exception as e:
        logger.error("Failed to load TTS model: %s", e)
        TTS_AVAILABLE = False
else:
    logger.warning("TTS dependencies not available; /tts will be disabled")

app = FastAPI(title="Vietnam Airlines Chatbot API (LangChain + Hybrid)")

# ---------------------
# LangChain integration
# ---------------------
qa_chain = None
if LANGCHAIN_AVAILABLE:
    try:
        # Create a simple LangChain-compatible retriever that wraps your query_vna_info
        class VNARetriever:
            def __init__(self, collection):
                self.collection = collection

            def get_relevant_documents(self, query: str):
                try:
                    content = query_vna_info(self.collection, query) if self.collection else ""
                except Exception as e:
                    logger.warning("vna retriever query failed: %s", e)
                    content = ""
                # Return a list of langchain Document objects (page_content only)
                return [Document(page_content=content)] if content else []

            async def aget_relevant_documents(self, query: str):
                return self.get_relevant_documents(query)

        retriever = VNARetriever(vna_collection)

        # Build the LangChain LLM wrapper for Azure
        llm = AzureChatOpenAI(
            azure_endpoint=_azure_endpoint,
            api_key=_azure_api_key,
            deployment_name=DEPLOYMENT,
            api_version="2024-07-01-preview",
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )

        # Small prompt template used by the RetrievalQA chain
        qa_prompt = ChatPromptTemplate.from_template(
            """You are an assistant that helps users with Vietnam Airlines information, booking flow, ticket prices, and onboarding.
Answer concisely in the requested language (English or Vietnamese). If the information is not present in the provided context, say you don't know and suggest contacting the airline.

Context:
{context}

User question:
{question}
"""
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt},
        )
        logger.info("LangChain RetrievalQA initialized and ready.")
    except Exception as e:
        logger.exception("Failed to initialize LangChain components: %s", e)
        qa_chain = None
else:
    logger.warning("LangChain not available in environment; running without LangChain.")

# ---------------------
# FastAPI endpoints
# ---------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.post("/chat")
async def chat(request: Request):
    """
    POST /chat
    Body JSON: { userId: string, message: string, language?: "en" | "vi", chatId?: string, fetchHistory?: bool }
    Returns: { reply: string, history?: [...] }
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_id = data.get("userId")
    message = data.get("message")
    chat_id = data.get("chatId") or "default"
    language = data.get("language", "en")
    fetch_history = bool(data.get("fetchHistory", False))

    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="userId (string) is required")
    if not fetch_history and (not message or not isinstance(message, str)):
        raise HTTPException(status_code=400, detail="message (string) is required")

    session_key = f"{user_id}::{chat_id}"
    session = SESSIONS.setdefault(session_key, [])

    if fetch_history and (not message or message.strip() == ""):
        return {"reply": "", "history": session}

    if message.strip().lower() in ("repeat last", "repeat", "nhắc lại", "lặp lại"):
        if not session:
            return {"reply": "No previous reply to repeat."}
        for m in reversed(session):
            if m["role"] == "assistant":
                return {"reply": m["content"]}
        return {"reply": "No previous reply to repeat."}

    # Append user message & persist
    session.append({"role": "user", "content": message})
    _save_sessions_to_disk()

    # Try to get relevant info from ChromaDB
    relevant_info = ""
    if vna_collection:
        try:
            relevant_info = query_vna_info(vna_collection, message) or ""
        except Exception as e:
            logger.warning("Failed to query ChromaDB: %s", e)

    # Build a system prompt consistent with previous behavior
    system_prompt = (
        "You are an assistant that helps users with Vietnam Airlines information, booking flow, ticket prices, and onboarding. "
        "Answer concisely in the requested language (English or Vietnamese). If the information is not on the official site, say you don't know and suggest contacting the airline. "
        "Base your answer on this relevant information from official Vietnam Airlines sources: " + relevant_info
    )

    # Build model messages (we keep last 10 messages)
    model_messages = [{"role": "system", "content": system_prompt}] + session[-10:]

    # Primary: try LangChain RetrievalQA (if available)
        # --- New: handle flight-related queries locally ---
    if any(kw in message.lower() for kw in ["flight", "ticket", "schedule", "route", "chuyến bay", "vé", "lịch trình"]):
        fake_results = query_fake_flights(message)
        if fake_results:
            reply = format_flight_results(fake_results)
            # Save assistant reply and persist, then return directly
            session.append({"role": "assistant", "content": reply})
            _save_sessions_to_disk()
            if fetch_history:
                return {"reply": reply, "history": session}
            return {"reply": reply}

    reply = ""
    if qa_chain is not None:
        try:
            # RetrievalQA.run expects a question string
            # We pass the user message; chain will call retriever internally.
            reply_candidate = await asyncio.to_thread(qa_chain.run, message)
            if reply_candidate and isinstance(reply_candidate, str) and reply_candidate.strip():
                reply = reply_candidate.strip()
        except Exception as e:
            logger.exception("LangChain QA failed, will fallback to direct AzureOpenAI call: %s", e)
            reply = ""

    # Fallback: direct AzureOpenAI call (original behavior)
    if not reply:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=DEPLOYMENT,
                messages=model_messages,
                max_tokens=MAX_TOKENS,
                temperature=0.2,
            )
        except RateLimitError as exc:
            logger.warning("Rate limited: %s", exc)
            raise HTTPException(status_code=429, detail="Rate limited; try again later.") from exc
        except (APITimeoutError, APIConnectionError) as exc:
            logger.error("OpenAI connection error: %s", exc)
            raise HTTPException(status_code=504, detail="Upstream service unavailable") from exc
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream API error") from exc
        except Exception as exc:
            logger.exception("Unexpected error calling model")
            raise HTTPException(status_code=500, detail="Unexpected error calling model") from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise HTTPException(status_code=502, detail="No response from model")

        # Extract reply from response shape (SDK dependent)
        try:
            reply = choices[0].message.content if hasattr(choices[0].message, "content") else getattr(choices[0].message, "content", "")
            reply = (reply or "").strip()
        except Exception:
            # last-resort: try to stringify response
            reply = str(response)[:1000]

    # Save assistant reply and persist
    session.append({"role": "assistant", "content": reply})
    _save_sessions_to_disk()

    if fetch_history:
        return {"reply": reply, "history": session}
    return {"reply": reply}

@app.get("/history")
async def get_history(userId: str, chatId: str = "default"):
    if not userId:
        raise HTTPException(status_code=400, detail="userId is required")
    session_key = f"{userId}::{chatId}"
    history = SESSIONS.get(session_key, [])
    return {"userId": userId, "chatId": chatId, "history": history}

@app.delete("/history")
async def delete_history(userId: str, chatId: str = "default"):
    if not userId:
        raise HTTPException(status_code=400, detail="userId is required")
    session_key = f"{userId}::{chatId}"
    if session_key in SESSIONS:
        del SESSIONS[session_key]
    _save_sessions_to_disk()
    return {"ok": True, "userId": userId, "chatId": chatId}

# TTS helpers & endpoint
def _synthesize_wav_bytes(text: str) -> bytes:
    if not TTS_AVAILABLE:
        raise RuntimeError("TTS model is not available on the server")

    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform

    if isinstance(output, torch.Tensor):
        arr = output.detach().cpu().numpy()
    else:
        arr = np.asarray(output)

    if arr.dtype.kind == 'f':
        maxv = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
        if maxv > 0:
            arr = arr / maxv

    if arr.ndim > 1:
        arr = arr.reshape(-1)

    buf = io.BytesIO()
    try:
        import soundfile as sf
        sf.write(buf, arr, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            from scipy.io.wavfile import write as wav_write
            if arr.dtype.kind == 'f':
                scaled = (arr * 32767).astype(np.int16)
            else:
                scaled = arr.astype(np.int16)
            wav_write(buf, SAMPLE_RATE, scaled)
            buf.seek(0)
            return buf.getvalue()
        except Exception as e:
            raise RuntimeError(f"No audio writer available on server: {e}")

@app.post("/tts")
async def tts_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    text = data.get("text")
    if not text or not isinstance(text, str):
        raise HTTPException(status_code=400, detail="text (string) is required")

    if not TTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="TTS not available on server")

    try:
        wav_bytes = await asyncio.to_thread(_synthesize_wav_bytes, text)
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=wav_bytes, media_type="audio/wav")

@app.get("/debug_openai")
async def debug_openai():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        raise HTTPException(status_code=500, detail="AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is not set in environment")

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=DEPLOYMENT,
            messages=[{"role": "system", "content": "Say 'ok'"}, {"role": "user", "content": "Hello"}],
            max_tokens=10,
            temperature=0.0,
        )
    except RateLimitError as exc:
        return JSONResponse(status_code=429, content={"ok": False, "detail": "rate_limited", "message": str(exc)})
    except (APITimeoutError, APIConnectionError) as exc:
        return JSONResponse(status_code=504, content={"ok": False, "detail": "connection_error", "message": str(exc)})
    except APIError as exc:
        return JSONResponse(status_code=502, content={"ok": False, "detail": "api_error", "message": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"ok": False, "detail": "unexpected", "message": str(exc)})

    choices = getattr(response, "choices", None)
    if not choices:
        return JSONResponse(status_code=502, content={"ok": False, "detail": "no_choices"})

    reply = choices[0].message.content if hasattr(choices[0].message, "content") else getattr(choices[0].message, "content", "")
    return {"ok": True, "detail": "success", "reply": (reply or "").strip()}

@app.post("/new_chat")
async def new_chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_id = data.get("userId")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=400, detail="userId (string) is required")

    chat_id = str(uuid.uuid4())
    session_key = f"{user_id}::{chat_id}"
    SESSIONS[session_key] = []
    _save_sessions_to_disk()
    return {"userId": user_id, "chatId": chat_id}
