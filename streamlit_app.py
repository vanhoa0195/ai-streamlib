import streamlit as st
import requests
import os
import json
from pathlib import Path
from base64 import b64encode
import numpy as np

BACKEND_URL = "http://3.25.83.174"
# BACKEND_URL = "http://localhost:3000"
STORE_PATH = Path(__file__).parent / "chats_store.json"

st.set_page_config(page_title="Vietnam Airlines Service Chatbot", layout="wide")

# ==============================
# 🌈 Custom CSS Styling
# ==============================
st.markdown("""
<style>
/* General Layout */
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    color: #1f2937;
}
.main {
    background-color: #f9fafb;
}

/* Title */
h1 {
    text-align: center;
    color: #0033A0;
    margin-bottom: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {

    color: white;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #FFD700;
}
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] select {
    background-color: white !important;
    color: #111 !important;
    border-radius: 8px;
}

/* Tabs */
div[data-testid="stTabs"] {
    background-color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Chat Section */
.chat-bubble-user {
    background-color: #E0E7FF;
    color: #1E3A8A;
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    margin-bottom: 10px;
    margin-left: auto;
    max-width: 80%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
.chat-bubble-bot {
    background-color: #F3F4F6;
    color: #111827;
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    margin-bottom: 10px;
    margin-right: auto;
    max-width: 80%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
}
.stButton>button {
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    transform: translateY(-2px);
}

/* Actions Sidebar */
.block-container {
    padding-top: 1rem;
}
div[data-testid="stVerticalBlock"] > div:has(h3) {
    margin-bottom: 1.2rem;
}
h3 {
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 0.3rem;
    color: #1E3A8A;
}
.section-box {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 5px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

/* File Upload + Download Buttons */
.stDownloadButton>button {
    background-color: #059669;
    color: white;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-weight: 600;
}
.stDownloadButton>button:hover {
    background-color: #047857;
}

/* Audio Button */
div[data-testid="stHorizontalBlock"] div[data-testid="column"] button {
    border: 1px solid #d1d5db;
    background-color: #fff;
    border-radius: 8px;
    color: #374151;
}
div[data-testid="stHorizontalBlock"] div[data-testid="column"] button:hover {
    border-color: #2563EB;
    color: #2563EB;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# PAGE HEADER
# ==============================
st.title("Vietnam Airlines Service Chatbot 🤖")
st.markdown("<hr style='margin-top:0; border:1px solid #ccc;'>", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")
    user_id = st.text_input("User ID", value="user1")
    language = st.selectbox("Language", options=["en - English", "vi - Vietnamese"], index=0)

# ==============================
# PREDEFINED QUESTIONS TAB
# ==============================
chat_tab, predefined_tab = st.tabs(["💬 Chat", "📝 Predefined Questions"])

with predefined_tab:
    st.header("Import Predefined Questions")
    st.write("Upload a `.txt` file with one question per line to test the chatbot quickly.")
    qfile = st.file_uploader("Upload file", type=["txt"])
    
    if qfile is not None:
        content = qfile.read().decode("utf-8")
        questions = [l.strip() for l in content.splitlines() if l.strip()]
        if questions:
            st.success(f"✅ Found {len(questions)} questions.")
            for q in questions:
                if st.button(f"Ask: {q}", key=f"pred_{q}") and not st.session_state.loading:
                    st.session_state.history.append({"role": "user", "content": q})
                    st.session_state.chats[st.session_state.active_chat] = st.session_state.history
                    save_chats_to_disk()
                    st.session_state.pending_send = q
                    st.experimental_rerun()
        else:
            st.warning("⚠️ No valid questions found in file.")

# ==============================
# STATE MANAGEMENT
# ==============================
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "pending_send" not in st.session_state:
        st.session_state.pending_send = None
    if "chats" not in st.session_state:
        if STORE_PATH.exists():
            try:
                with open(STORE_PATH, "r", encoding="utf-8") as _f:
                    st.session_state.chats = json.load(_f)
            except Exception:
                st.session_state.chats = {"default": []}
        else:
            st.session_state.chats = {"default": []}

init_session_state()
if "active_chat" not in st.session_state:
    st.session_state.active_chat = "default"

# ==============================
# FUNCTIONS
# ==============================
def send_message(msg: str):
    payload = {"userId": user_id, "message": msg, "language": language, "chatId": st.session_state.active_chat}
    try:
        r = requests.post(BACKEND_URL + "/chat", json=payload, timeout=30)
    except Exception as e:
        return f"Error contacting backend: {e}"
    if r.status_code != 200:
        try:
            return f"Error {r.status_code}: {r.json().get('error', r.text)}"
        except Exception:
            return f"Error {r.status_code}: {r.text}"
    return r.json().get("reply", "")

def fetch_history_from_server(user_id: str, chat_id: str):
    try:
        r = requests.get(BACKEND_URL + "/history", params={"userId": user_id, "chatId": chat_id}, timeout=10)
        if r.status_code == 200:
            return r.json().get("history", [])
    except Exception:
        return []
    return []

def save_chats_to_disk():
    try:
        with open(STORE_PATH, "w", encoding="utf-8") as _f:
            json.dump(st.session_state.chats, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Initial history load
if "initial_fetch_done" not in st.session_state:
    st.session_state.initial_fetch_done = False
if not st.session_state.initial_fetch_done:
    try:
        srv_hist = fetch_history_from_server(user_id, st.session_state.active_chat)
        if srv_hist:
            st.session_state.chats[st.session_state.active_chat] = srv_hist
            st.session_state.history = srv_hist
            save_chats_to_disk()
    except Exception:
        pass
    st.session_state.initial_fetch_done = True

# ==============================
# CHAT LOGIC
# ==============================
if st.session_state.get("pending_send"):
    msg_to_send = st.session_state.pop("pending_send")
    st.session_state.loading = True
    with st.spinner("✈️ Bot is responding..."):
        reply = send_message(msg_to_send)
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.chats[st.session_state.active_chat] = st.session_state.history
    save_chats_to_disk()
    st.session_state.loading = False

# ==============================
# CHAT TAB
# ==============================
with chat_tab:
    chat_container, actions_container = st.columns([3, 1])
    
    # Chat History
    with chat_container:
        st.subheader("💬 Chat")
        for idx, item in enumerate(st.session_state.history):
            if item["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {item['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-bot'><b>Bot:</b> {item['content']}</div>", unsafe_allow_html=True)
                if st.button("🔊", key=f"tts_{idx}", help="Click to listen"):
                    try:
                        r = requests.post(BACKEND_URL + "/tts", json={"text": item['content']}, timeout=60)
                        if r.status_code == 200:
                            audio_str = f"data:audio/wav;base64,{b64encode(r.content).decode()}"
                            st.markdown(f"""
                                <audio autoplay style="display:none">
                                    <source src="{audio_str}" type="audio/wav">
                                </audio>
                            """, unsafe_allow_html=True)
                    except Exception:
                        pass

        # Chat Input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", key="input")
            if st.form_submit_button("Send") and user_input and not st.session_state.loading:
                st.session_state.history.append({"role": "user", "content": user_input})
                st.session_state.chats[st.session_state.active_chat] = st.session_state.history
                save_chats_to_disk()
                st.session_state.pending_send = user_input
                st.experimental_rerun()

    # ==============================
    # ACTIONS SIDEBAR
    # ==============================
    with actions_container:
        st.subheader("⚙️ Actions")
        st.markdown("### 💭 Conversations")
        chat_select, new_chat = st.columns([2, 1])
        with chat_select:
            chat_ids = list(st.session_state.chats.keys())
            selected = st.selectbox("Active chat", options=chat_ids)
            if selected != st.session_state.active_chat:
                st.session_state.active_chat = selected
                srv_hist = fetch_history_from_server(user_id, selected)
                st.session_state.history = srv_hist or st.session_state.chats.get(selected, [])
                save_chats_to_disk()
                st.experimental_rerun()
        with new_chat:
            if st.button("New Chat"):
                new_id = f"chat{len(st.session_state.chats)+1}"
                st.session_state.chats[new_id] = []
                st.session_state.active_chat = new_id
                st.session_state.history = []
                save_chats_to_disk()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 🧭 Controls")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                srv_hist = fetch_history_from_server(user_id, st.session_state.active_chat)
                if srv_hist:
                    st.session_state.chats[st.session_state.active_chat] = srv_hist
                    st.session_state.history = srv_hist
                    save_chats_to_disk()
                    st.experimental_rerun()
        with col2:
            if st.session_state.active_chat != "default" and st.button("🗑️ Delete"):
                try:
                    requests.delete(
                        BACKEND_URL + "/history",
                        params={"userId": user_id, "chatId": st.session_state.active_chat},
                        timeout=10
                    )
                except Exception:
                    pass
                st.session_state.chats.pop(st.session_state.active_chat, None)
                st.session_state.active_chat = "default"
                st.session_state.history = st.session_state.chats.get("default", [])
                save_chats_to_disk()
                st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("### 💾 Export")
        data_json = json.dumps(
            st.session_state.chats.get(st.session_state.active_chat, []),
            ensure_ascii=False,
            indent=2
        )
        st.download_button(
            "📥 Download Chat",
            data=data_json,
            file_name=f"chat_{user_id}_{st.session_state.active_chat}.json",
            mime="application/json"
        )
        st.markdown("</div>", unsafe_allow_html=True)
