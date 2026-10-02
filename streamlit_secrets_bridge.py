# This file is loaded before anything else by Streamlit.
# It bridges st.secrets → os.environ so all dotenv-based code works unchanged.
import os
import streamlit as st

def _inject_secrets_to_env():
    """Push Streamlit secrets into os.environ so load_dotenv()-based code works."""
    secret_keys = [
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ]
    for key in secret_keys:
        try:
            value = st.secrets.get(key) or st.secrets.get("general", {}).get(key)
            if value and key not in os.environ:
                os.environ[key] = value
        except Exception:
            pass  # Will fall back to .env / existing env vars


_inject_secrets_to_env()
