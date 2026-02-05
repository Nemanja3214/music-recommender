import json
import os

import requests

from playlist_iterator import iter_mpd_playlists, LIMIT
from mongo import MongoCache


def get_artist(artist_id: str):
    url = f"https://api.spotify.com/v1/artists/{artist_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers, timeout=20)
    print(response.status_code)
    # exit()
    return response.json()

def get_spotify_artists(track_spotify_id):
    tr = sp.track(track_spotify_id)
    artists = []

    for artist in tr["artists"]:
        try:
            art_obj = get_artist(artist["id"])
        except:
            continue
        for g in art_obj.get("genres", []):
            art_obj["genres"] = g.lower().replace(" ", "_")
        artists.append(art_obj)
        print(art_obj)
        time.sleep(1)

    return artists

if __name__ == "__main__":

    # -------------------------------
    # Example usage
    # -------------------------------
    mpd_dir = "./archive/data"

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
    sp = spotipy.Spotify(auth=access_token, requests_timeout=20)
    cache = MongoCache()


    playlists = []
    for pl in iter_mpd_playlists(mpd_dir):
        playlists.append(pl)
        if LIMIT and len(playlists) >= LIMIT:
            break

    unique = {}
    for pl in playlists:
        for t in pl['tracks']:
            unique[t['track_uri']] = t
    tids = list(unique.keys())

    import signal
    import time


    for tid in tids:
        if cache.exists(tid, cache.ARTISTS_NAME):
            print("kesirano")
            continue
        track_id = tid.split(":")[-1]
        artists = get_spotify_artists(track_id)

        cache.add(tid, cache.ARTISTS_NAME, {"artists": artists})





