"""Streamlit UI + modular BiLSTM + eliza.

This module depends on:
- scripts/chat.py
"""

import os
import sys

# project root must be on sys.path before importing scripts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import random

import streamlit as st
from scripts.chat import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_STOPWORDS_PATH,
    STREAMLIT_CHAT_CSS,
    chat_turn,
    load_chat_context,
    resolve_project_path,
)


# function to get environment variable or return default
def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# get environment variables or use defaults
CHECKPOINT = resolve_project_path(_env("CHAT_CHECKPOINT", DEFAULT_CHECKPOINT_PATH))
STOPWORDS = resolve_project_path(_env("CHAT_STOPWORDS", DEFAULT_STOPWORDS_PATH))
LANGUAGE = _env("CHAT_LANGUAGE", "mm")


# function to load chat context
@st.cache_resource
def _cached_ctx(checkpoint_path: str, language: str):
    lang = language if language in ("mm", "en") else "mm"
    return load_chat_context(checkpoint_path, language=lang)


# set page config and CSS
st.set_page_config(page_title="Group 2 — Hybrid ELIZA", layout="centered")
st.markdown(STREAMLIT_CHAT_CSS, unsafe_allow_html=True)

ctx = _cached_ctx(CHECKPOINT, LANGUAGE)

# load eliza
eliza = ctx["eliza"]

# display title and caption
st.title("Group 2 — Hybrid ELIZA")
st.caption("Rule-based ELIZA powered by Burmese NLP model")

# initialize messages and greeted state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

if not st.session_state.greeted:
    greeting = random.choice(eliza.script["initials"])
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.greeted = True

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# display messages
if prompt := st.chat_input("စာသား ရိုက်ပါ …"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    out = chat_turn(ctx, prompt, STOPWORDS)

    if out["kind"] == "empty":
        st.stop()

    if out["kind"] == "quit":
        assistant_text = f"Eliza: {out['final']}"
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.stop()

    assistant_text = (
        f"Predicted emotion: {out['emotion_label']} ({out['emotion_score']:.2%})\n\n"
        f"Eliza: {out['eliza_reply']}"
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_text)
    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
