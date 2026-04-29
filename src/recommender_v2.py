from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import re

_LOW_ENERGY_WORDS = {
    "calm", "chill", "relaxed", "mellow", "soft", "quiet", "sleep", "study", "lounge"
}
_HIGH_ENERGY_WORDS = {
    "energetic", "upbeat", "dance", "party", "hype", "workout", "drive", "intense", "powerful"
}
_HIGH_VALENCE_WORDS = {
    "happy", "bright", "sunny", "joy", "uplifting", "positive", "cheerful", "optimistic", "glad"
}
_LOW_VALENCE_WORDS = {
    "sad", "dark", "moody", "melancholic", "angry", "broody", "somber", "lonely"
}
_ACOUSTIC_WORDS = {
    "acoustic", "organic", "unplugged", "folk", "natural", "stripped"
}
_ELECTRONIC_WORDS = {
    "electronic", "synth", "synthwave", "produced", "digital", "beat", "edm", "trance", "techno"
}

@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float = 0.0
    liveness: float = 0.0
    popularity: float = 0.0
    description: str = ""

    def document_text(self) -> str:
        text_parts = [
            self.title,
            self.artist,
            self.genre,
            self.mood,
            self.description,
        ]
        return " ".join(part.strip() for part in text_parts if part).lower()

@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    valence: float = 0.5


def _get_value(song, key, default=None):
    if isinstance(song, dict):
        return song.get(key, default)
    return getattr(song, key, default)


def _song_to_dict(song):
    if isinstance(song, dict):
        return song
    return asdict(song)


def _normalize_text(song) -> str:
    if isinstance(song, dict):
        parts = [
            song.get("title", ""),
            song.get("artist", ""),
            song.get("genre", ""),
            song.get("mood", ""),
            song.get("description", ""),
        ]
        return " ".join(part.strip() for part in parts if part).lower()
    return song.document_text()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def load_songs(csv_path: str) -> List[Song]:
    songs: List[Song] = []
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Songs CSV not found: {csv_path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row:
                continue

            try:
                song = Song(
                    id=int(row.get("id", 0) or 0),
                    title=row.get("title", "").strip(),
                    artist=row.get("artist", "").strip(),
                    genre=row.get("genre", "").strip(),
                    mood=row.get("mood", "").strip(),
                    energy=float(row.get("energy", 0.0) or 0.0),
                    tempo_bpm=float(row.get("tempo_bpm", 0.0) or 0.0),
                    valence=float(row.get("valence", 0.0) or 0.0),
                    danceability=float(row.get("danceability", 0.0) or 0.0),
                    acousticness=float(row.get("acousticness", 0.0) or 0.0),
                    instrumentalness=float(row.get("instrumentalness", 0.0) or 0.0),
                    liveness=float(row.get("liveness", 0.0) or 0.0),
                    popularity=float(row.get("popularity", 0.0) or 0.0),
                    description=row.get("description", "").strip(),
                )
            except ValueError:
                continue

            songs.append(song)
    return songs


def _retrieval_score(query: str, song) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    document_text = _normalize_text(song)
    document_tokens = set(_tokenize(document_text))
    overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)

    score = overlap * 0.55

    if query_tokens & _LOW_ENERGY_WORDS:
        score += (1.0 - _get_value(song, "energy", 0.5)) * 0.20
    elif query_tokens & _HIGH_ENERGY_WORDS:
        score += _get_value(song, "energy", 0.5) * 0.20

    if query_tokens & _HIGH_VALENCE_WORDS:
        score += _get_value(song, "valence", 0.5) * 0.10
    elif query_tokens & _LOW_VALENCE_WORDS:
        score += (1.0 - _get_value(song, "valence", 0.5)) * 0.10

    if query_tokens & _ACOUSTIC_WORDS:
        score += _get_value(song, "acousticness", 0.5) * 0.10
    elif query_tokens & _ELECTRONIC_WORDS:
        score += (1.0 - _get_value(song, "acousticness", 0.5)) * 0.10

    return min(score, 1.0)


def retrieve_candidates(query: str, songs: List, n: int = 20) -> List:
    if not query:
        return songs[:n]

    ranked = sorted(
        [(song, _retrieval_score(query, song)) for song in songs],
        key=lambda item: item[1],
        reverse=True,
    )
    return [song for song, _ in ranked[:n]]


def score_song(user_prefs: Dict, song) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    song_genre = _get_value(song, "genre", "")
    song_mood = _get_value(song, "mood", "")

    if user_prefs.get("mood") and user_prefs.get("mood") == song_mood:
        score += 0.35
        reasons.append("mood match (+0.35)")

    energy_contribution = (
        1.0 - abs(user_prefs.get("energy", 0.5) - _get_value(song, "energy", 0.5))
    ) * 0.25
    score += energy_contribution
    reasons.append(f"energy proximity (+{energy_contribution:.2f})")

    if user_prefs.get("genre") and user_prefs.get("genre") == song_genre:
        score += 0.20
        reasons.append("genre match (+0.20)")

    acousticness = _get_value(song, "acousticness", 0.5)
    acoustic_raw = acousticness if user_prefs.get("likes_acoustic", True) else (1.0 - acousticness)
    acoustic_contribution = acoustic_raw * 0.12
    score += acoustic_contribution
    reasons.append(f"acoustic fit (+{acoustic_contribution:.2f})")

    valence_contribution = (
        1.0 - abs(user_prefs.get("valence", 0.5) - _get_value(song, "valence", 0.5))
    ) * 0.08
    score += valence_contribution
    reasons.append(f"valence proximity (+{valence_contribution:.2f})")

    score = max(0.0, min(score, 1.0))
    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List, k: int = 5, query: str = "") -> List[Tuple[Dict, float, str]]:
    candidate_songs = retrieve_candidates(query, songs, n=max(k * 4, 20)) if query else songs
    scored = [
        (song, *score_song(user_prefs, song))
        for song in candidate_songs
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    return [(_song_to_dict(song), score, ", ".join(reasons)) for song, score, reasons in ranked[:k]]
