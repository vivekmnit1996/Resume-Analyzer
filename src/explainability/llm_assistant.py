"""
explainability/llm_assistant.py
Grounded LLM explanation and career advising layer.

Key Architectural Decisions:
1. Grounded Context (RAG-style grounding):
   - The LLM does NOT retrieve or recommend jobs.
   - The ML recommendation engine computes top jobs, match scores, matched skills,
     and skill gaps.
   - This structured context is formatted and injected into the LLM system prompt.
2. Guardrails:
   - The model is instructed to answer strictly based on the provided candidates and skills.
   - If asked about details not present in the data, it explicitly acknowledges the limitation.
3. Security:
   - API key is retrieved via os.getenv("GROQ_API_KEY") or Streamlit secrets.
   - Never hardcoded.
   - Graceful fallback when API key is unset or network fails.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests as http_requests

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_api_key() -> Optional[str]:
    """Retrieves Groq API key from environment or Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            pass
    return key


def format_grounded_context(
    resume_profile: Any,
    top_jobs: Any
) -> str:
    """
    Constructs a dense, structured string representation of ML recommendation results.
    """
    context_lines = ["=== CANDIDATE PROFILE ==="]
    if hasattr(resume_profile, 'skills_raw'):
        context_lines.append(f"Detected Skills: {', '.join(resume_profile.skills_raw[:25])}")
    if hasattr(resume_profile, 'years_of_experience') and resume_profile.years_of_experience is not None:
        context_lines.append(f"Estimated Experience: {resume_profile.years_of_experience} years")
    if hasattr(resume_profile, 'education') and resume_profile.education:
        context_lines.append(f"Education: {', '.join(resume_profile.education)}")

    context_lines.append("\n=== TOP RECOMMENDED JOBS (RANKED BY ML ENGINE) ===")
    
    if hasattr(top_jobs, 'iterrows'):
        for i, (_, row) in enumerate(top_jobs.head(5).iterrows(), start=1):
            title = row.get('job_title', 'Unknown Title')
            cat = row.get('category', 'Unknown Category')
            score = row.get('ranking_score_pct', row.get('similarity_pct', 0.0))
            matched = row.get('matched_skills', [])
            missing = row.get('missing_skills', [])

            context_lines.append(
                f"[Job #{i}] {title} ({cat}) - Match Score: {score}%\n"
                f"  Matched Skills: {', '.join(matched) if matched else 'None recorded'}\n"
                f"  Missing Required Skills: {', '.join(missing) if missing else 'None detected'}\n"
            )

    return "\n".join(context_lines)


def build_grounded_system_prompt(grounded_context: str) -> str:
    return f"""You are the Intelligent Career Assistant for a Resume Matching system.
Your job is to provide clear, actionable career advice based STRICTLY on the candidate's actual resume and the ML system's top job matches.

GROUNDED SYSTEM CONTEXT:
{grounded_context}

RULES:
1. Ground your answers strictly in the provided resume profile, match scores, matched skills, and missing skills.
2. Do NOT invent or fabricate job titles, companies, or requirements not listed in the context.
3. When asked "Why was this job recommended?", refer directly to the match score, overlapping skills, and semantic fit.
4. When asked what to learn or improve, focus on the specific missing skills from the top jobs.
5. If the user asks a question that cannot be answered from the provided data, politely say that the information is unavailable from the resume or current postings.
6. Keep answers concise, professional, and structured with bullet points.
"""


def chat_grounded_assistant(
    messages: List[Dict[str, str]],
    resume_profile: Any = None,
    top_jobs: Any = None
) -> str:
    """
    Sends chat history to Groq LLaMA with injected recommendation grounding.
    """
    api_key = get_groq_api_key()
    if not api_key or api_key == "your_groq_api_key_here":
        return (
            "⚠️ **Groq API Key Not Configured**\n\n"
            "To use the interactive LLaMA assistant, set the `GROQ_API_KEY` environment variable in `.env` "
            "or Streamlit secrets.\n\n"
            "All core ML features (skills extraction, semantic search, hybrid retrieval, ranking, "
            "and rule-based explanations) are fully operational offline without any API key."
        )

    # Build grounded system message
    grounded_context = format_grounded_context(resume_profile, top_jobs)
    system_prompt = build_grounded_system_prompt(grounded_context)

    # Prepare message payload
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": full_messages,
        "max_tokens": 1500,
        "temperature": 0.4  # Lower temperature for grounded factual consistency
    }

    try:
        resp = http_requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
        return f"Groq API returned status {resp.status_code}: {resp.text[:150]}"
    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        return "Connection error while reaching Groq API. Please verify network access and API credentials."
