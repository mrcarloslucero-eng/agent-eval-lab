import os
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Reddit blocks anonymous .json access (HTTP 403) from most IPs, so the tools
# prefer OAuth client-credentials tokens when REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET
# are set in .env and fall back to a plain anonymous request otherwise.
UA = os.getenv("REDDIT_USER_AGENT", "AgentEvalLab/1.0")
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

_token_cache = {"value": None}


def _oauth_token() -> str | None:
    if _token_cache["value"]:
        return _token_cache["value"]
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None
    try:
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": UA},
            timeout=15,
        )
        if response.status_code == 200:
            _token_cache["value"] = response.json().get("access_token")
    except requests.RequestException:
        pass
    return _token_cache["value"]


def _reddit_get(path: str) -> requests.Response:
    token = _oauth_token()
    if token:
        return requests.get(
            f"https://oauth.reddit.com{path}",
            headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
            timeout=15,
        )
    return requests.get(
        f"https://www.reddit.com{path}",
        headers={"User-Agent": BROWSER_UA},
        timeout=15,
    )


def _error_message(status: int, sub: str) -> str:
    hint = ""
    if status == 403:
        hint = " Anonymous .json access is blocked from this network; add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env."
    return f"Error: Reddit returned {status} for r/{sub}.{hint}"


def _format_posts(posts: list, sub: str) -> str:
    if not posts:
        return f"No posts found in r/{sub}."
    lines = []
    for i, p in enumerate(posts, start=1):
        title = str(p.get("title", "?"))[:80]
        author = p.get("author", "?")
        lines.append(f"- #{i} [{title}] by u/{author} (score {p.get('score', 0)}, {p.get('num_comments', 0)} comments)")
    return "\n".join(lines)


def _fetch_posts(sub: str, path: str, limit: int) -> str:
    sub = sub.removeprefix("r/").strip("/")
    limit = max(1, min(limit, 10))
    response = _reddit_get(f"{path}&limit={limit}&raw_json=1")
    if response.status_code != 200:
        return _error_message(response.status_code, sub)
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        return (f"Error: Reddit returned a non-JSON page (HTTP {response.status_code}) for r/{sub}."
                " Anonymous access is likely blocked; add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env.")
    children = payload.get("data", {}).get("children", [])
    posts = [c["data"] for c in children if c.get("kind") == "t3"]
    return _format_posts(posts[:limit], sub)


# -------- TOOLS --------

@tool
def get_hot_posts(subreddit: str, limit: int = 5) -> str:
    """Get the hottest posts from a subreddit. Format: 'LocalLLaMA' or 'r/LocalLLaMA'."""
    return _fetch_posts(subreddit, f"/r/{subreddit.removeprefix('r/').strip('/')}/hot?", limit)


@tool
def search_posts(subreddit: str, query: str, limit: int = 5) -> str:
    """Search posts within a subreddit. Format: subreddit 'LocalLLaMA', query free text."""
    sub = subreddit.removeprefix("r/").strip("/")
    return _fetch_posts(sub, f"/r/{sub}/search?q={quote(query)}&restrict_sr=1&sort=relevance?", limit)


# -------- AGENT --------

model = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")
tools = [get_hot_posts, search_posts]
agent = create_react_agent(model, tools)

def run_agent(query: str):
    """Run the agent and return the full message trace."""
    result = agent.invoke({"messages": [("user", query)]})
    return result
