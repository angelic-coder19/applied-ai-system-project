"""Tests for the conflict-aware recommender in recommender_v2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommender_v2 import query_conflict_penalty, recommend_songs


def test_query_conflict_penalty_detects_mixed_energy():
    penalty = query_conflict_penalty("calm party")
    assert penalty > 0.0
    assert penalty <= 0.35


def test_query_conflict_penalty_is_zero_for_consistent_requests():
    assert query_conflict_penalty("energetic rock music") == 0.0


def test_recommend_songs_applies_conflict_penalty_to_scores():
    songs = [
        {
            "id": 1,
            "title": "Dance Through Midnight",
            "artist": "Neon Pulse",
            "genre": "electronic",
            "mood": "intense",
            "energy": 0.95,
            "tempo_bpm": 130,
            "valence": 0.75,
            "danceability": 0.88,
            "acousticness": 0.10,
        }
    ]
    prefs = {"genre": "electronic", "mood": "intense", "energy": 0.95, "likes_acoustic": False, "valence": 0.75}

    result_without_conflict = recommend_songs(prefs, songs, k=1, query="intense dance")
    result_with_conflict = recommend_songs(prefs, songs, k=1, query="intense dance calm")

    score_without = result_without_conflict[0][1]
    score_with = result_with_conflict[0][1]
    explanation_with = result_with_conflict[0][2]

    assert score_with < score_without
    assert "confidence penalty" in explanation_with
