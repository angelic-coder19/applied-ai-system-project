from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.recommender_v2 import query_conflict_penalty, recommend_songs

assert query_conflict_penalty('calm party') > 0.0
assert query_conflict_penalty('energetic rock music') == 0.0

songs = [
    {
        'id': 1,
        'title': 'Dance Through Midnight',
        'artist': 'Neon Pulse',
        'genre': 'electronic',
        'mood': 'intense',
        'energy': 0.95,
        'tempo_bpm': 130.0,
        'valence': 0.75,
        'danceability': 0.88,
        'acousticness': 0.10,
    }
]

prefs = {
    'genre': 'electronic',
    'mood': 'intense',
    'energy': 0.95,
    'likes_acoustic': False,
    'valence': 0.75,
}

without = recommend_songs(prefs, songs, k=1, query='intense dance')
with_conflict = recommend_songs(prefs, songs, k=1, query='intense dance calm')
assert with_conflict[0][1] < without[0][1]
assert 'confidence penalty' in with_conflict[0][2]
print('OK')
