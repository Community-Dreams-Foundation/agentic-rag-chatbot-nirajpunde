"""Selective memory: extract high-signal facts and append to USER_MEMORY.md / COMPANY_MEMORY.md."""

import json
import os
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


MEMORY_SYSTEM_PROMPT = """You decide whether to store high-signal facts from a conversation.

Rules:
- Store ONLY reusable, high-signal facts (e.g., "User is a Project Finance Analyst", "Prefers weekly summaries on Mondays").
- Do NOT store: raw transcripts, PII, secrets, trivial chitchat.
- Target USER for user-specific facts (role, preferences, workflows).
- Target COMPANY for org-wide learnings (interfaces, bottlenecks, patterns useful to colleagues).
- Be selective. Only output when confident (confidence >= 0.7).

Respond with a JSON object or array. Each item: {"should_write": true, "target": "USER"|"COMPANY", "summary": "...", "confidence": 0.0-1.0}
If nothing worth storing: {"should_write": false}
"""

MEMORY_USER_TEMPLATE = """Conversation excerpt:
User: {user_message}
Assistant: {assistant_message}

Any high-signal fact to store? Output JSON only."""


def get_memory_llm():
    """Create Gemini model for memory extraction."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def _parse_memory_response(text: str) -> list[dict]:
    """Parse LLM response into list of {target, summary} for writing."""
    results = []
    text = text.strip()
    # Try to extract JSON
    try:
        # Handle single object
        if text.startswith("{"):
            obj = json.loads(text)
            if obj.get("should_write") and obj.get("target") and obj.get("summary"):
                conf = obj.get("confidence", 0)
                if conf >= 0.7:
                    results.append({"target": obj["target"], "summary": obj["summary"]})
        # Handle array
        elif text.startswith("["):
            arr = json.loads(text)
            for item in arr:
                if isinstance(item, dict) and item.get("should_write") and item.get("target") and item.get("summary"):
                    if item.get("confidence", 0) >= 0.7:
                        results.append({"target": item["target"], "summary": item["summary"]})
    except json.JSONDecodeError:
        pass
    return results


def extract_and_write_memory(user_message: str, assistant_message: str, base_dir: Path = None) -> list[dict]:
    """
    Run memory extraction and append to USER_MEMORY.md / COMPANY_MEMORY.md.
    Returns list of {"target": "USER"|"COMPANY", "summary": "..."} that were written.
    """
    base = base_dir or Path(__file__).resolve().parent.parent
    user_mem_path = base / "USER_MEMORY.md"
    company_mem_path = base / "COMPANY_MEMORY.md"

    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SYSTEM_PROMPT),
        ("human", MEMORY_USER_TEMPLATE),
    ])
    llm = get_memory_llm()
    chain = prompt | llm
    response = chain.invoke({
        "user_message": user_message,
        "assistant_message": assistant_message,
    })
    text = response.content if hasattr(response, "content") else str(response)
    items = _parse_memory_response(text)

    written = []
    for item in items:
        target = item["target"]
        summary = item["summary"]
        path = user_mem_path if target == "USER" else company_mem_path
        line = f"- [{datetime.now().strftime('%Y-%m-%d')}] {summary}\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        written.append({"target": target, "summary": summary})

    return written
