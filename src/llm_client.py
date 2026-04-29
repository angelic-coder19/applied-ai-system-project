"""
LLM client for Beats Buddy RAG system.

Two responsibilities:
  1. parse_preferences  — convert natural language query → structured preference dict
  2. generate_recommendation — convert retrieved songs + query → conversational response
"""

import json
import logging
import os
from typing import Dict, List

try:
    from src.env_loader import load_dotenv
except ImportError:
    from env_loader import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_VALID_GENRES = {
    "pop", "lofi", "rock", "ambient", "jazz", "synthwave",
    "indie pop", "hip hop", "classical", "metal", "country",
    "reggae", "electronic", "folk", "rnb", "blues", "world",
}
_VALID_MOODS = {
    "happy", "chill", "intense", "moody", "focused", "relaxed",
    "energetic", "melancholic", "aggressive", "nostalgic", "dreamy",
    "playful", "romantic", "sultry", "somber", "uplifting",
}
_DEFAULT_PREFS: Dict = {
    "genre": "pop",
    "mood": "chill",
    "energy": 0.5,
    "likes_acoustic": False,
    "valence": 0.5,
}

_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")


def _get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )
    from google import genai  # imported lazily so missing package gives a clear error
    return genai.Client(api_key=api_key)


def parse_preferences(user_query: str) -> Dict:
    """
    Use an LLM to extract structured music preferences from a natural language query.

    Returns a dict with keys: genre, mood, energy, likes_acoustic, valence.
    Falls back to sensible defaults if the LLM call fails.
    """
    prompt = (
        "You are a music preference parser. Given a user description of what music they want, "
        "extract their preferences and return ONLY a JSON object with exactly these keys:\n"
        "- genre (string, one of: pop, lofi, rock, ambient, jazz, synthwave, indie pop, "
        "hip hop, classical, metal, country, reggae, electronic, folk, rnb, blues, world)\n"
        "- mood (string, one of: happy, chill, intense, moody, focused, relaxed, energetic, "
        "melancholic, aggressive, nostalgic, dreamy, playful, romantic, sultry, somber, uplifting)\n"
        "- energy (float 0.0–1.0, where 0=very calm, 1=very energetic)\n"
        "- likes_acoustic (boolean, true=organic/acoustic, false=produced/electronic)\n"
        "- valence (float 0.0–1.0, where 0=sad/dark, 1=happy/bright)\n\n"
        f'User query: "{user_query}"\n\n'
        "Return ONLY valid JSON. No markdown, no explanation."
    )

    try:
        client = _get_client()
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        raw = response.text.strip()

        # Strip markdown code fences if the model wrapped its output
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        prefs = json.loads(raw)

        # Validate categorical fields; clamp numerics
        if prefs.get("genre") not in _VALID_GENRES:
            prefs["genre"] = _DEFAULT_PREFS["genre"]
        if prefs.get("mood") not in _VALID_MOODS:
            prefs["mood"] = _DEFAULT_PREFS["mood"]
        prefs["energy"] = max(0.0, min(1.0, float(prefs.get("energy", 0.5))))
        prefs["likes_acoustic"] = bool(prefs.get("likes_acoustic", False))
        prefs["valence"] = max(0.0, min(1.0, float(prefs.get("valence", 0.5))))

        logger.info("Parsed preferences: %s", prefs)
        return prefs

    except EnvironmentError:
        raise
    except Exception as exc:
        logger.warning("Preference parsing failed (%s); using defaults.", exc)
        return dict(_DEFAULT_PREFS)


def generate_recommendation(user_query: str, songs: List[Dict], low_confidence: bool = False) -> str:
    """
    Generate a conversational recommendation using the retrieved songs.

    songs must be a list of dicts each containing at least:
      title, artist, genre, mood, score (float)
    """
    song_lines = "\n".join(
        f"- {s['title']} by {s['artist']} "
        f"(genre: {s['genre']}, mood: {s['mood']}, match score: {s['score']:.2f})"
        for s in songs
    )
    prompt = (
        "You are a friendly music recommendation assistant called Beats Buddy.\n"
        f'A user asked: "{user_query}"\n\n'
        "The recommendation engine retrieved these top matching songs from the catalog:\n"
        f"{song_lines}\n\n"
        "Write a short, conversational response (3–5 sentences) that:\n"
        "1. Recommends the top 2–3 songs by their exact titles\n"
        "2. Explains WHY each one fits this specific request (mood, energy, genre)\n"
        "3. Sounds warm and natural, like a knowledgeable friend — not a bullet list\n\n"
        "Do not just list all five songs. Focus on the best fits and say why."
    )

    if low_confidence:
        prompt += (
            "\nThe user request contains mixed or conflicting signals, and the retrieved matches should be treated as lower-confidence suggestions. "
            "Do not use strong endorsement language like 'you'll love this' or 'perfectly captures.' "
            "Instead, say that these are the best approximate options available, that the request is hard to satisfy exactly, "
            "and phrase the recommendation cautiously."
        )

    try:
        client = _get_client()
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        result = response.text.strip()
        logger.info("Generated recommendation (%d chars).", len(result))
        return result

    except EnvironmentError:
        raise
    except Exception as exc:
        logger.warning("Recommendation generation failed (%s); falling back to list.", exc)
        return "Here are your top matches:\n" + "\n".join(
            f"  - {s['title']} by {s['artist']}" for s in songs
        )
