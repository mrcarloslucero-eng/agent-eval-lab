import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def setup_tracing():
    """Call this before running any agent to enable LangSmith tracing."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    # LANGCHAIN_API_KEY and LANGCHAIN_PROJECT are picked up from .env automatically
    print(f"✅ Tracing enabled for project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")