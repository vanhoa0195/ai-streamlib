import streamlit as st
import requests
import os
import json
from pathlib import Path
import io
import importlib
import numpy as np
from base64 import b64encode

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")
STORE_PATH = Path(__file__).parent / "chats_store.json"

st.set_page_config(page_title="Vietnam Airlines Service Chatbot", layout="wide")

st.title("Vietnam Airlines Service Chatbot")

# Move global settings to sidebar
with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="user1")
    language = st.selectbox("Language", options=["en - English", "vi - Vietnamese"], index=0)

# Create tabs for chat and predefined questions
chat_tab, predefined_tab = st.tabs(["💬 Chat", "📝 Predefined Questions"])

# Handle predefined questions tab first so chat tab gets priority in UI updates
with predefined_tab:
    st.header("Import Predefined Questions")
    st.write("Upload a text file containing questions (one per line) to test the chatbot with predefined scenarios.")
    qfile = st.file_uploader("Upload questions file", type=["txt"])
    
    if qfile is not None:
        content = qfile.read().decode("utf-8")
        questions = [l.strip() for l in content.splitlines() if l.strip()]
        if questions:
            st.write(f"Found {len(questions)} questions in file")
            for q in questions:
                if st.button(f"Ask: {q}", key=f"pred_{q}") and not st.session_state.loading:
                    # Append user message and schedule it to be sent in the next run
                    st.session_state.history.append({"role": "user", "content": q})
                    st.session_state.chats[st.session_state.active_chat] = st.session_state.history
                    save_chats_to_disk()
                    st.session_state.pending_send = q
                    # Try to switch to chat tab visually by triggering rerun (programmatic tab switch isn't supported reliably)
                    st.experimental_rerun()
        else:
            st.warning("No questions found in the uploaded file")

# Initialize session state
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "pending_send" not in st.session_state:
        st.session_state.pending_send = None
    if "chats" not in st.session_state:
        # chats is a dict chatId -> list of messages
        # Try to load from disk
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

# On first page load, automatically fetch history for the active chat from the
# backend to improve UX (so users see server-side history without clicking
# "Refresh from server"). We guard with a session flag so this runs only once
# per browser session.


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


# On first page load, automatically fetch history for the active chat from the
# backend to improve UX (so users see server-side history without clicking
# "Refresh from server"). We guard with a session flag so this runs only once
# per browser session. Placed after helper defs so functions are available.
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
        # don't block the UI on failure; fall back to local cache
        pass
    st.session_state.initial_fetch_done = True

# If there's a pending message to send (set in the previous run), process it now.
if st.session_state.get("pending_send"):
    msg_to_send = st.session_state.pop("pending_send")
    # Ensure the UI knows we're loading
    st.session_state.loading = True
    with st.spinner("Bot is typing..."):
        reply = send_message(msg_to_send)
    # Append assistant reply and persist
    st.session_state.history.append({"role": "assistant", "content": reply})
    st.session_state.chats[st.session_state.active_chat] = st.session_state.history
    save_chats_to_disk()
    st.session_state.loading = False

# Main chat interface
with chat_tab:
    chat_container, actions_container = st.columns([3, 1])
    
    with chat_container:
        st.subheader("Chat")
        # Chat history display
        for idx, item in enumerate(st.session_state.history):
            if item["role"] == "user":
                st.markdown(f"**You:** {item['content']}")
            else:
                # Create a row for bot message with speaker icon
                msg_container = st.container()
                with msg_container:
                    msg_col, btn_col = st.columns([20, 1])
                    with msg_col:
                        st.markdown(f"**Bot:** {item['content']}")
                    with btn_col:
                        # Style the button to look more modern
                        unique_key = f"tts_btn_{st.session_state.active_chat}_{idx}"
                        audio_state_key = f"audio_state_{st.session_state.active_chat}_{idx}"
                        
                        # Initialize audio state in session if not exists
                        if audio_state_key not in st.session_state:
                            st.session_state[audio_state_key] = {"audio_data": None}
                        
                        # Custom CSS for the button
                        st.markdown("""
                            <style>
                            div[data-testid="stHorizontalBlock"] div[data-testid="column"] button {
                                padding: 0px 12px;
                                border: 1px solid #e0e0e0;
                                background-color: transparent;
                            }
                            div[data-testid="stHorizontalBlock"] div[data-testid="column"] button:hover {
                                border-color: #1e88e5;
                                color: #1e88e5;
                            }
                            /* Remove default Streamlit button styles */
                            div[data-testid="stHorizontalBlock"] div[data-testid="column"] button::after {
                                display: none !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        if st.button("🔊", key=unique_key, help="Click to listen"):
                            try:
                                r = requests.post(BACKEND_URL + "/tts", json={"text": item['content']}, timeout=60)
                                if r.status_code == 200:
                                    # Play audio immediately and invisibly
                                    audio_str = f"data:audio/wav;base64,{b64encode(r.content).decode()}"
                                    st.markdown(f"""
                                        <audio autoplay style="display:none">
                                            <source src="{audio_str}" type="audio/wav">
                                        </audio>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            except Exception:
                                pass  # Silently fail to avoid UI disruption

        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", key="input")
            submit = st.form_submit_button("Send")
            if submit and user_input and not st.session_state.loading:
                # Append the user's message and schedule it to be sent in the next run.
                message = {"role": "user", "content": user_input}
                st.session_state.history.append(message)
                st.session_state.chats[st.session_state.active_chat] = st.session_state.history
                save_chats_to_disk()
                # Use pending_send so the next script run sends the message and appends the assistant reply.
                st.session_state.pending_send = user_input
                try:
                    st.session_state["input"] = ""
                except Exception:
                    pass
                # Trigger a rerun so the pending_send is processed immediately
                st.experimental_rerun()

    with actions_container:
        st.subheader("Actions")
        # Chat management
        st.markdown("### Conversations")
        chat_select, new_chat = st.columns([2, 1])
        
        with chat_select:
            chat_ids = list(st.session_state.chats.keys())
            selected = st.selectbox(
                "Active chat",
                options=chat_ids,
                index=chat_ids.index(st.session_state.active_chat) if st.session_state.active_chat in chat_ids else 0
            )
            if selected != st.session_state.active_chat:
                st.session_state.active_chat = selected
                # Attempt to fetch latest history from server when switching chats
                srv_hist = fetch_history_from_server(user_id, selected)
                if srv_hist:
                    st.session_state.chats[selected] = srv_hist
                    st.session_state.history = srv_hist
                else:
                    # Fallback to local cached history
                    st.session_state.history = st.session_state.chats.get(selected, [])
                save_chats_to_disk()
                st.experimental_rerun()

        with new_chat:
            if st.button("📝 New"):
                new_id = f"chat{len(st.session_state.chats)+1}"
                st.session_state.chats[new_id] = []
                st.session_state.active_chat = new_id
                st.session_state.history = []
                save_chats_to_disk()
                st.experimental_rerun()

        # Chat Controls
        st.markdown("### Controls")
        control_cols = st.columns(2)
        
        with control_cols[0]:
            if st.button("🔄 Refresh") and not st.session_state.loading:
                srv_hist = fetch_history_from_server(user_id, st.session_state.active_chat)
                if srv_hist:
                    st.session_state.chats[st.session_state.active_chat] = srv_hist
                    st.session_state.history = srv_hist
                    save_chats_to_disk()
                    st.experimental_rerun()
        
        with control_cols[1]:
            if st.button("↩️ Repeat Last") and not st.session_state.loading:
                # Append the user command and schedule it for the next run
                message = {"role": "user", "content": "repeat last"}
                st.session_state.history.append(message)
                st.session_state.chats[st.session_state.active_chat] = st.session_state.history
                save_chats_to_disk()
                st.session_state.pending_send = "repeat last"
                st.experimental_rerun()

        # Delete current chat
        if st.session_state.active_chat != "default":
            if st.button("🗑️ Delete Chat"):
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

        # Export section
        st.markdown("### Export")
        import json as _json
        data_json = _json.dumps(
            st.session_state.chats.get(st.session_state.active_chat, []),
            ensure_ascii=False,
            indent=2
        )
        st.download_button(
            "💾 Download Chat",
            data=data_json,
            file_name=f"chat_{user_id}_{st.session_state.active_chat}.json",
            mime="application/json"
        )


