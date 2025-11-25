import json
import os
import sqlite3
import time
from typing import Optional, List, Tuple

def iter_mpd_playlists(mpd_dir: str, n=None):
    """Yield playlists from each .json file in mpd_dir (sorted)."""
    files = [os.path.join(mpd_dir, f) for f in os.listdir(mpd_dir) if f.endswith('.json')]
    if n is None:
        files = sorted(files)
    else:
        files = sorted(files)[:n]

    print(f"There is {len(files)} files")
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as fh:
            try:
                data = json.load(fh)
            except Exception as e:
                print(f'Warning: failed to load {fp}: {e}')
                continue
            for pl in data.get('playlists', []):
                yield pl


def get_spotify_artists_and_genres(track_spotify_id, cache):
    genres = set()
    try:
        tr = sp.track(track_spotify_id)
    except:
        return genres

    for artist in tr["artists"]:
        try:
            art_obj = sp.artist(artist["id"])
        except:
            continue
        for g in art_obj.get("genres", []):
            genres.add(g.lower().replace(" ", "_"))
    genres = list(genres)
    print(tr["artists"], genres)
    # print(cache)

    return tr["artists"], genres

if __name__ == "__main__":

    # -------------------------------
    # Example usage
    # -------------------------------
    mpd_dir = "./archive/data"
    cache_path = "artist_cache.json"
    limit = 2000

    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    with open("api_keys", "r") as f:
        lines = [line.strip() for line in f.readlines()]

    CLIENT_ID = lines[0]
    CLIENT_SECRET = lines[1]
    REDIRECT_URI = lines[2]

    sp_oauth = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope="user-library-read playlist-modify-public"
    )

    # Get cached token or prompt authorization
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        print(f"Please open this URL in your browser to authorize: {auth_url}")
        response = input("Enter the full URL you were redirected to: ")
        code = sp_oauth.parse_response_code(response)
        token_info = sp_oauth.get_access_token(code)

    access_token = token_info['access_token']
    sp = spotipy.Spotify(auth=access_token)
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as fh:
                cache = json.load(fh)
                print(f"Cache has {len(cache)} items" )
        except Exception:
            cache = {}


    playlists = []
    for pl in iter_mpd_playlists(mpd_dir, 1):
        playlists.append(pl)
        if limit and len(playlists) >= limit:
            break

    unique = {}
    for pl in playlists:
        for t in pl['tracks']:
            unique[t['track_uri']] = t
    tids = list(unique.keys())

    import signal
    import time

    def cleanup_handler(signum, frame):
        print("Handling sigkill")
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
        exit(0) # Exit gracefully

    signal.signal(signal.SIGTERM, cleanup_handler)

    try:
        for tid in tids:
            if tid in cache:
                # print("kesirano")
                continue
            track_id = tid.split(":")[-1]
            artists, genres = get_spotify_artists_and_genres(track_id, cache)
            time.sleep(1)
            cache[tid] = {
                "artists": artists,
                "genres": genres
            }
    except KeyboardInterrupt as e:
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)
    except Exception as e:
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh)





