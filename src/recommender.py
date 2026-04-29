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

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs: List[Dict] = []

    print(f"Loading songs from {csv_path}...")
    path = Path(csv_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Songs CSV not found: {csv_path}")
    
        
    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Normalize and convert types
            try:
                song = {
                    'id': int(row.get('id', '').strip()) if row.get('id') is not None and row.get('id').strip() != '' else None,
                    'title': row.get('title', '').strip(),
                    'artist': row.get('artist', '').strip(),
                    'genre': row.get('genre', '').strip(),
                    'mood': row.get('mood', '').strip(),
                    'energy': float(row.get('energy', 0.0)) if row.get('energy') not in (None, '') else 0.0,
                    'tempo_bpm': float(row.get('tempo_bpm', 0.0)) if row.get('tempo_bpm') not in (None, '') else 0.0,
                    'valence': float(row.get('valence', 0.0)) if row.get('valence') not in (None, '') else 0.0,
                    'danceability': float(row.get('danceability', 0.0)) if row.get('danceability') not in (None, '') else 0.0,
                    'acousticness': float(row.get('acousticness', 0.0)) if row.get('acousticness') not in (None, '') else 0.0,
                }
            except ValueError:
                # Skip rows with bad numeric values
                continue

            songs.append(song)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Returns a weighted score in [0.0, 1.0] and a list of human-readable reasons.

    Weights: mood 0.35 | energy 0.25 | genre 0.20 | acousticness 0.12 | valence 0.08
    """
    score = 0.0
    reasons: List[str] = []

    # Mood match (weight: 0.35) — highest weight, strongest intent signal
    if user_prefs.get("mood") and user_prefs.get("mood") == song.get("mood"):
        score += 0.35
        reasons.append("mood match (+0.35)")

    # Energy proximity (weight: 0.25) — how close the song's energy is to the target
    energy_contribution = (1.0 - abs(user_prefs.get("energy", 0.5) - song.get("energy", 0.5))) * 0.25
    score += energy_contribution
    reasons.append(f"energy proximity (+{energy_contribution:.2f})")

    # Genre match (weight: 0.20) — penalized less than mood due to sparse catalog
    if user_prefs.get("genre") and user_prefs.get("genre") == song.get("genre"):
        score += 0.20
        reasons.append("genre match (+0.20)")

    # Acoustic direction score (weight: 0.12) — direction flip based on boolean preference
    acousticness = song.get("acousticness", 0.5)
    acoustic_raw = acousticness if user_prefs.get("likes_acoustic", True) else (1.0 - acousticness)
    acoustic_contribution = acoustic_raw * 0.12
    score += acoustic_contribution
    reasons.append(f"acoustic fit (+{acoustic_contribution:.2f})")

    # Valence proximity (weight: 0.08) — emotional positiveness, kept low to avoid mood overlap
    valence_contribution = (1.0 - abs(user_prefs.get("valence", 0.5) - song.get("valence", 0.5))) * 0.08
    score += valence_contribution
    reasons.append(f"valence proximity (+{valence_contribution:.2f})")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Ranks all songs using score_song as the judge and returns the top k.
    Required by src/main.py
    """
    scored = sorted(
        [(song, *score_song(user_prefs, song)) for song in songs],
        key=lambda item: item[1],
        reverse=True
    )
    return [(song, score, ", ".join(reasons)) for song, score, reasons in scored[:k]]
