"""Streamlit Cloud / single-command entry: imports the streamlit chatter module after putting the project root on sys.path."""

# import streamlit chatter module to run Streamlit UI at import time
from scripts.streamlit_chatter import render_app

# render app
render_app()