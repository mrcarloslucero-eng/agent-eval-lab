import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# -------- TOOLS --------

@tool
def get_commits(repo: str, limit: int = 5) -> str:
    """Get the latest commits for a GitHub repository. Format: 'owner/repo'"""
    url = f"https://api.github.com/repos/{repo}/commits?per_page={limit}"
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return f"Error: GitHub returned {response.status_code} for repo '{repo}'"
    
    commits = response.json()
    lines = []
    for c in commits:
        sha = c.get("sha", "?")[:7]
        author = c.get("commit", {}).get("author", {}).get("name", "Unknown")
        msg = c.get("commit", {}).get("message", "No message").split("\n")[0][:60]
        lines.append(f"- {sha} by {author}: {msg}")
    return "\n".join(lines) if lines else "No commits found."

@tool
def get_issues(repo: str, label: str = "bug", limit: int = 5) -> str:
    """Get open issues for a GitHub repository with a specific label. Format: 'owner/repo'"""
    url = f"https://api.github.com/repos/{repo}/issues?state=open&labels={label}&per_page={limit}"
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return f"Error: GitHub returned {response.status_code} for repo '{repo}'"
    
    issues = response.json()
    if not issues:
        return f"No open issues with label '{label}' found in {repo}."
    
    lines = []
    for i in issues:
        num = i.get("number", "?")
        title = i.get("title", "No title")
        user = i.get("user", {}).get("login", "Unknown")
        lines.append(f"- #{num} {title} (by {user})")
    return "\n".join(lines)

# -------- AGENT --------

model = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")  # Adjust base_url if needed
tools = [get_commits, get_issues]
agent = create_react_agent(model, tools)

def run_agent(query: str):
    """Run the agent and return the full message trace."""
    result = agent.invoke({"messages": [("user", query)]})
    return result