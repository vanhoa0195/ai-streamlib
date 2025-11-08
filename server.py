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

DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "GPT-4o-mini")
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

def extract_dates_from_text(text: str) -> List[datetime]:
    """
    Extract multiple dates from query with enhanced support for various formats and expressions.
    Returns a list of datetime objects.
    """
    text = text.lower()
    dates = []
    today = datetime.now()

    # Handle relative date references in both English and Vietnamese
    relative_dates = {
        "today": 0, "hôm nay": 0, "homnay": 0,
        "tomorrow": 1, "ngày mai": 1, "ngaymai": 1,
        "next week": 7, "tuần sau": 7, "tuansau": 7,
        "next month": 30, "tháng sau": 30, "thangsau": 30
    }

    # Check for relative dates
    for phrase, days in relative_dates.items():
        if phrase in text:
            dates.append(today + timedelta(days=days))

    # Handle date ranges
    range_patterns = {
        "en": [
            r"between (\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})\s+(?:and|to)\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4})",
            r"from (\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) to (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ],
        "vi": [
            r"từ (\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) đến (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"giữa (\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) và (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ]
    }

    for patterns in range_patterns.values():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    start_date = parse_date_string(match.group(1))
                    end_date = parse_date_string(match.group(2))
                    if start_date and end_date:
                        dates.extend([start_date, end_date])
                except:
                    continue

    # Handle flexible dates
    flexible_patterns = {
        "en": [
            r"(?:around|about|approximately) (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) \(?±\d+ days?\)?"
        ],
        "vi": [
            r"(?:khoảng|gần|xung quanh) (\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}) \(?±\d+ ngày\)?"
        ]
    }

    for patterns in flexible_patterns.values():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    base_date = parse_date_string(match.group(1))
                    if base_date:
                        dates.append(base_date)
                except:
                    continue

    # Handle specific date formats
    date_patterns = [
        # Standard formats
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",
        # Month name formats (English)
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{2,4})",
        # Vietnamese date formats
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+(?:năm\s+)?(\d{2,4})?",
        # Abbreviated Vietnamese
        r"(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?"
    ]

    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                date_parts = match.groups()
                if len(date_parts) == 3:
                    year = int(date_parts[2]) if date_parts[2] else today.year
                    if year < 100:
                        year += 2000
                    month = int(date_parts[1])
                    day = int(date_parts[0])
                    dates.append(datetime(year, month, day))
            except:
                continue

    # Remove duplicates while preserving order
    seen = set()
    unique_dates = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    return unique_dates

def parse_date_string(date_str: str) -> Optional[datetime]:
    """Helper function to parse various date string formats."""
    try:
        # Try various date formats
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
            "%d/%m/%y", "%d-%m-%y",
            "%b %d %Y", "%d %b %Y",
            "%B %d %Y", "%d %B %Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    except:
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
    """Trả về bản tóm tắt chuyến bay dễ đọc cho trợ lý."""
    if not flights:
        return "Xin lỗi, tôi không tìm thấy chuyến bay nào phù hợp với yêu cầu của bạn."

    msg_lines = ["Dưới đây là một số chuyến bay của Vietnam Airlines mà bạn có thể quan tâm:\n"]
    for f in flights:
        msg_lines.append(
            f"✈️ **{f['flight_no']}** — {f['origin']} → {f['destination']} "
            f"({f['aircraft']}, {f['duration_hr']} giờ)\n"
            f"🕓 Khởi hành: {f['departure']} | Hạ cánh: {f['arrival']} | Tần suất: {f.get('weekly_frequency','Hàng ngày')}\n"
            f"💺 Tình trạng: {f['status']}\n"
            f"💰 Giá vé: Phổ thông {f['prices']['Economy']:,}₫, "
            f"Phổ thông đặc biệt {f['prices']['Premium Economy']:,}₫, Thương gia {f['prices']['Business']:,}₫\n"
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
        # --- Handle flight-related queries using the enhanced flight API ---
    if any(kw in message.lower() for kw in ["flight", "ticket", "schedule", "route", "chuyến bay", "vé", "lịch trình"]):
        # Determine language based on message content
        language = "vi" if any(word in message.lower() for word in ["chuyến", "vé", "lịch"]) else "en"
        
        # Extract route information
        route_info = {}
        city_codes = {
            # Vietnamese cities with variations
            "hanoi": "HAN", "ha noi": "HAN", "hà nội": "HAN",
            "ho chi minh": "SGN", "saigon": "SGN", "sài gòn": "SGN", "tphcm": "SGN",
            "da nang": "DAD", "đà nẵng": "DAD", "danang": "DAD",
            "nha trang": "CXR",
            "phu quoc": "PQC", "phú quốc": "PQC",
            "hue": "HUI", "huế": "HUI",
            "can tho": "VCA", "cần thơ": "VCA",
            # International cities
            "tokyo": "NRT",
            "seoul": "ICN",
            "bangkok": "BKK",
            "singapore": "SIN",
            "kuala lumpur": "KUL",
            "melbourne": "MEL",
            "sydney": "SYD",
            "paris": "CDG",
            "london": "LHR",
            "frankfurt": "FRA"
        }
        
        message_lower = message.lower()
        # Try to find origin and destination
        for city, code in city_codes.items():
            if city in message_lower:
                if 'origin' not in route_info:
                    route_info['origin'] = code
                elif 'destination' not in route_info and code != route_info.get('origin'):
                    route_info['destination'] = code
                    break
        
        # Try to extract dates with the new enhanced function
        dates = extract_dates_from_text(message)
        if dates:
            route_info['depart_date'] = dates[0].strftime("%Y-%m-%d")
            if len(dates) > 1:  # If multiple dates found
                route_info['return_date'] = dates[1].strftime("%Y-%m-%d")
                # If more dates, assume multi-city
                if len(dates) > 2:
                    route_info['multi_city_dates'] = [d.strftime("%Y-%m-%d") for d in dates[2:]]
                    
        # Extract price range preferences
        price_ranges = {
            "budget": {"min": 0, "max": 2000000},  # Up to 2M VND
            "moderate": {"min": 2000000, "max": 5000000},  # 2M-5M VND
            "premium": {"min": 5000000, "max": float('inf')}  # Above 5M VND
        }
        
        price_keywords = {
            "en": {
                "budget": ["cheap", "budget", "economical", "inexpensive"],
                "moderate": ["moderate", "medium", "standard"],
                "premium": ["premium", "luxury", "expensive", "business"]
            },
            "vi": {
                "budget": ["rẻ", "tiết kiệm", "giá rẻ", "phổ thông"],
                "moderate": ["vừa phải", "trung bình", "tiêu chuẩn"],
                "premium": ["cao cấp", "sang trọng", "đắt", "thương gia"]
            }
        }
        
        # Detect price range preference
        for range_type, keywords in price_keywords[language].items():
            if any(kw in message_lower for kw in keywords):
                route_info['price_range'] = price_ranges[range_type]
                break
                
        # Extract flight preferences
        preferences = {
            "direct_only": any(kw in message_lower for kw in 
                             ["direct", "non-stop", "trực tiếp", "không dừng"]),
            "preferred_airlines": [],
            "meal_preference": None,
            "seat_preference": None
        }
        
        # Detect meal preferences
        meal_keywords = {
            "en": {
                "VEGETARIAN": ["vegetarian", "vegan"],
                "HALAL": ["halal"],
                "KOSHER": ["kosher"],
                "DIABETIC": ["diabetic", "diabetes"]
            },
            "vi": {
                "VEGETARIAN": ["chay", "ăn chay"],
                "HALAL": ["halal"],
                "KOSHER": ["kosher"],
                "DIABETIC": ["tiểu đường"]
            }
        }
        
        for meal_type, keywords in meal_keywords[language].items():
            if any(kw in message_lower for kw in keywords):
                preferences['meal_preference'] = meal_type
                break
                
        # Detect seat preferences
        seat_keywords = {
            "en": {
                "WINDOW": ["window"],
                "AISLE": ["aisle"],
                "EXTRA_LEGROOM": ["extra leg", "more space"]
            },
            "vi": {
                "WINDOW": ["cửa sổ"],
                "AISLE": ["lối đi"],
                "EXTRA_LEGROOM": ["thêm chỗ", "rộng rãi"]
            }
        }
        
        for seat_type, keywords in seat_keywords[language].items():
            if any(kw in message_lower for kw in keywords):
                preferences['seat_preference'] = seat_type
                break
        
        # Extract passenger counts
        passenger_info = {
            'adult_count': 1,  # Default to 1 adult
            'child_count': 0,
            'infant_count': 0
        }
        
        # Look for number patterns followed by passenger type indicators
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'năm': 5
        }
        
        # Extract numeric values
        import re
        numbers = re.findall(r'\d+', message)
        words = message_lower.split()
        
        for i, word in enumerate(words):
            if word in number_words or word.isdigit():
                num = int(word) if word.isdigit() else number_words[word]
                # Check next word for passenger type
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if any(typ in next_word for typ in ['adult', 'người lớn', 'nguoi lon']):
                        passenger_info['adult_count'] = num
                    elif any(typ in next_word for typ in ['child', 'trẻ em', 'tre em']):
                        passenger_info['child_count'] = num
                    elif any(typ in next_word for typ in ['infant', 'em bé', 'em be']):
                        passenger_info['infant_count'] = num
        
        # Detect cabin class preference
        cabin_class = "ECONOMY"  # Default
        if any(cls in message_lower for cls in ["business", "thương gia", "thuong gia"]):
            cabin_class = "BUSINESS"
        elif any(cls in message_lower for cls in ["premium", "đặc biệt", "dac biet"]):
            cabin_class = "PREMIUM_ECONOMY"
        
        # If we have sufficient information, try the API
        if route_info.get('origin') and route_info.get('destination') and route_info.get('depart_date'):
            try:
                from flight_api import search_flights
                # Determine sorting criteria based on user preferences
                sort_criteria = "price"  # Default
                if any(kw in message_lower for kw in ["earliest", "sớm nhất"]):
                    sort_criteria = "departure"
                elif any(kw in message_lower for kw in ["shortest", "fastest", "nhanh nhất", "ngắn nhất"]):
                    sort_criteria = "duration"
                elif any(kw in message_lower for kw in ["stops", "quá cảnh"]):
                    sort_criteria = "stops"
                
                # Build search parameters
                search_params = {
                    "start_point": route_info['origin'],
                    "end_point": route_info['destination'],
                    "depart_date": route_info['depart_date'],
                    "return_date": route_info.get('return_date'),
                    "adult_count": passenger_info['adult_count'],
                    "child_count": passenger_info['child_count'],
                    "infant_count": passenger_info['infant_count'],
                    "cabin_class": cabin_class,
                    "language": language,
                    "sort_by": sort_criteria
                }
                
                # Add multi-city information if available
                if route_info.get('multi_city_dates'):
                    search_params['multi_city_dates'] = route_info['multi_city_dates']
                
                # Add price range if specified
                if 'price_range' in route_info:
                    search_params['min_price'] = route_info['price_range']['min']
                    search_params['max_price'] = route_info['price_range']['max']
                
                # Add preferences if specified
                if preferences.get('direct_only'):
                    search_params['direct_only'] = True
                if preferences.get('meal_preference'):
                    search_params['meal_preference'] = preferences['meal_preference']
                if preferences.get('seat_preference'):
                    search_params['seat_preference'] = preferences['seat_preference']
                
                flight_results = search_flights(**search_params
                )
                
                if flight_results and 'formatted_response' in flight_results:
                    reply = flight_results['formatted_response']
                else:
                    # Fallback messages based on language
                    reply = ("No flights found for your criteria. Please try different dates or routes." 
                            if language == "en" else 
                            "Không tìm thấy chuyến bay phù hợp. Vui lòng thử các ngày hoặc tuyến đường khác.")
                
                session.append({"role": "assistant", "content": reply})
                _save_sessions_to_disk()
                if fetch_history:
                    return {"reply": reply, "history": session}
                return {"reply": reply}
                
            except Exception as e:
                logger.warning(f"Flight API failed: {e}")
                # Fallback message based on language
                reply = ("Sorry, I couldn't search for flights at the moment. Please try again later." 
                        if language == "en" else 
                        "Xin lỗi, hiện tại không thể tìm kiếm chuyến bay. Vui lòng thử lại sau.")
                session.append({"role": "assistant", "content": reply})
                _save_sessions_to_disk()
                if fetch_history:
                    return {"reply": reply, "history": session}
                return {"reply": reply}
        
        else:
            # Ask for missing information based on language
            missing_info = []
            if 'origin' not in route_info:
                missing_info.append("departure city" if language == "en" else "thành phố khởi hành")
            if 'destination' not in route_info:
                missing_info.append("destination city" if language == "en" else "thành phố đến")
            if 'depart_date' not in route_info:
                missing_info.append("travel date" if language == "en" else "ngày đi")
            
            if language == "en":
                reply = f"Please provide the following information to search for flights: {', '.join(missing_info)}"
            else:
                reply = f"Vui lòng cung cấp thông tin sau để tìm chuyến bay: {', '.join(missing_info)}"
            
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
