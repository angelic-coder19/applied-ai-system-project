"""
Command line runner for Beats Buddy — RAG Edition.

Default mode: natural language → LLM parses preferences → scoring algorithm
retrieves top songs → LLM generates a conversational recommendation.

Legacy mode: python src/main.py --profile STUDY|POP|ROCK
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from .recommender import load_songs, recommend_songs
    from .llm_client import parse_preferences, generate_recommendation
except ImportError:
    from recommender import load_songs, recommend_songs
    from llm_client import parse_preferences, generate_recommendation

PROFILES = {
    "STUDY": {
        "genre": "lofi",
        "mood": "focused",
        "energy": 0.42,
        "likes_acoustic": True,
        "valence": 0.58,
    },
    "POP": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.70,
        "likes_acoustic": False,
        "valence": 0.22,
    },
    "ROCK": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.90,
        "likes_acoustic": False,
        "valence": 0.46,
    },
}

W = 54
DIV = "-" * W


def _print_profile_results(user_prefs: dict, songs: list, recommendations: list, label: str) -> None:
    acoustic_label = "Yes" if user_prefs["likes_acoustic"] else "No"
    print(f"\n{DIV}")
    print("  BEATS BUDDY 2.0")
    print(DIV)
    print(f"\n  {label}")
    print(f"  {'Genre':<10}: {user_prefs['genre']:<14}  {'Acoustic':<10}: {acoustic_label}")
    print(f"  {'Mood':<10}: {user_prefs['mood']:<14}  {'Valence':<10}: {user_prefs['valence']}")
    print(f"  {'Energy':<10}: {user_prefs['energy']}")
    print()
    print(DIV)
    print(f"  Top {len(recommendations)} Recommendations  ({len(songs)} songs scored)")
    print(DIV)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        bar = "#" * int(score * 20)
        print(f"\n  #{rank}  {song['title']}  -  {song['artist']}")
        print(f"       Score  : {score:.2f}  [{bar:<20}]")
        print(f"       Genre  : {song['genre']:<12}  Mood : {song['mood']}")
        print(f"       Why    :", end="")
        for i, reason in enumerate(explanation.split(", ")):
            prefix = " " if i == 0 else " " * 15
            print(f"{prefix}{reason}")
    print(f"\n{DIV}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beats Buddy — AI Music Recommender")
    parser.add_argument(
        "--profile",
        choices=["STUDY", "POP", "ROCK"],
        help="Run a preset profile instead of natural language mode.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    songs_path = str(project_root / "data" / "songs.csv")

    if args.profile:
        # Legacy profile mode — no LLM calls, original behavior preserved
        user_prefs = PROFILES[args.profile]
        songs = load_songs(songs_path)
        recommendations = recommend_songs(user_prefs, songs, k=5)
        logger.info("Profile mode '%s' — retrieved: %s", args.profile, [s["title"] for s, _, _ in recommendations])
        _print_profile_results(user_prefs, songs, recommendations, f"Profile: {args.profile}")
        return

    # ── RAG Mode ─────────────────────────────────────────────────────────────
    print(f"\n{DIV}")
    print("  BEATS BUDDY 2.0  — AI Music Recommender")
    print(f"{DIV}")
    print("  Tell me what you're in the mood for and I'll find")
    print("  the perfect songs from our catalog.")
    print(f"{DIV}\n")

    user_query = input("  What kind of music do you want right now?\n  > ").strip()
    if not user_query:
        print("  No input received. Exiting.")
        sys.exit(0)

    logger.info("User query: %r", user_query)

    # Step 1 — LLM parses natural language into structured preference fields
    print("\n  Parsing your request with AI...\n")
    try:
        user_prefs = parse_preferences(user_query)
    except EnvironmentError as exc:
        print(f"\n  Setup error: {exc}")
        sys.exit(1)

    acoustic_label = "Yes" if user_prefs["likes_acoustic"] else "No"
    print(f"{DIV}")
    print("  Interpreted Preferences")
    print(f"  {'Genre':<10}: {user_prefs['genre']:<14}  {'Acoustic':<10}: {acoustic_label}")
    print(f"  {'Mood':<10}: {user_prefs['mood']:<14}  {'Valence':<10}: {user_prefs['valence']}")
    print(f"  {'Energy':<10}: {user_prefs['energy']}")

    # Step 2 — Retrieval: existing scoring algorithm ranks the catalog
    songs = load_songs(songs_path)
    recommendations = recommend_songs(user_prefs, songs, k=5)
    logger.info("Retrieved songs: %s", [s["title"] for s, _, _ in recommendations])

    print(f"\n{DIV}")
    print(f"  Top {len(recommendations)} Matches  ({len(songs)} songs scored)")
    print(DIV)
    for rank, (song, score, _) in enumerate(recommendations, start=1):
        bar = "#" * int(score * 20)
        print(f"\n  #{rank}  {song['title']}  -  {song['artist']}")
        print(f"       Score  : {score:.2f}  [{bar:<20}]")
        print(f"       Genre  : {song['genre']:<12}  Mood : {song['mood']}")

    # Step 3 — Generation: LLM writes a conversational response using the retrieved songs
    print(f"\n{DIV}")
    print("  Beats Buddy says...")
    print(DIV)

    songs_for_llm = [
        {**song, "score": score}
        for song, score, _ in recommendations
    ]
    try:
        ai_response = generate_recommendation(user_query, songs_for_llm)
    except EnvironmentError as exc:
        print(f"\n  Setup error: {exc}")
        sys.exit(1)

    print()
    for line in ai_response.splitlines():
        print(f"  {line}")
    print(f"\n{DIV}\n")


if __name__ == "__main__":
    main()
