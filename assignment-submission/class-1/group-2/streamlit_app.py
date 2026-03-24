"""Streamlit Cloud / single-command entry: imports the streamlit chatter module after putting the project root on sys.path."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# import streamlit chatter module to run Streamlit UI at import time
import scripts.streamlit_chatter