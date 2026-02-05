import json
import os
import ssl
from typing import Dict, List

from mongo import MongoCache

import os, certifi

from playlist_iterator import iter_mpd_playlists, LIMIT

os.environ["SSL_CERT_FILE"] = certifi.where()

# Cell 2: MPD iterator & small sample fallback

def spotify_url_to_uri(url: str) -> str:
    # Example: https://open.spotify.com/track/6GIrIt2M39wEGwjCQjGChX
    parts = url.rstrip("/").split("/")
    item_type = parts[-2]
    item_id = parts[-1].split("?")[0]
    return f"spotify:{item_type}:{item_id}"


import http.client

def augment_tracks(playlists: List[Dict], cache: MongoCache, batch_size=40):
    """Return dict track_uri -> feature dict for AUDIO_FEATURE_KEYS. Use cache if available."""
    unique = {}
    print("started augmenting tracks")
    artist_tracks_ids = list(cache.get_all_ids(cache.ARTISTS_NAME))
    artist_tracks_ids = [dictionary["_id"] for dictionary in artist_tracks_ids]
    print(artist_tracks_ids)

    for pl in playlists:
        for t in pl['tracks']:
            if t["track_uri"] in artist_tracks_ids:
                unique[t['track_uri']] = t

    print(f"Unique has {len(unique)}")
    tids = list(unique.keys())
    features = {}
    unfound = 0
    found = 0

    def chunk_list(original_list, chunk_size=batch_size):
        result_list = []
        # Iterate from index 0 to the end of the list, in steps of chunk_size
        for i in range(0, len(original_list), chunk_size):
            # Slice the original list and append the sublist to the result
            result_list.append(original_list[i:i + chunk_size])
        return result_list

    batch_tids = chunk_list(tids)

    def af_to_vec(af):
        if af is None:
            return {k: 0.0 for k in AUDIO_FEATURE_KEYS}
        return {k: float(af.get(k, 0.0)) for k in AUDIO_FEATURE_KEYS}

    for one_chunk_tids in batch_tids:
        # conn.request("GET", f'/v1/audio-features?ids={track_id}', payload, headers)
        url = '/v1/audio-features?ids='
        for i, tid in enumerate(one_chunk_tids):
            track_id = tid.split(":")[-1]  # take the last part after ":"

            print(tid)
            if cache.exists(tid, cache.TRACKS_NAME):
                features[tid] = cache.get(tid, cache.TRACKS_NAME)
                print("kesirano")
                # break
                continue

            url += str(track_id) + ","

        if url[-1] == ",":
            url = url[:-1]
        # print(url)
        conn.request("GET", url, payload, headers)
        # print(f'https://api.reccobeats.com/v1/audio-features?ids={track_id}')
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        data = json.loads(data)
        if "content" in data and len(data["content"]) > 0:
            data_arr = data["content"]
            # print(data_arr)
            if len(data_arr) > 0:
                for data_dict in data_arr:
                    tid = spotify_url_to_uri(data_dict["href"])
                    track_id = tid.split("/")[-1]
                    # print(tid)
                    af = data_dict
                    vec = af_to_vec(af)
                    features[track_id] = vec
                    cache.add(tid, cache.TRACKS_NAME, vec)

            found = len(data_arr)
            print(f"Founds: {found}")
            print(f"Unfounds: {40 - found}")
    return features

if __name__ == '__main__':
    mongo_cache = MongoCache()
    AUDIO_FEATURE_KEYS = [
        'danceability', 'energy', 'valence', 'tempo', 'loudness',
        'speechiness', 'instrumentalness', 'liveness', 'acousticness'
    ]

    context = ssl.create_default_context(cafile=certifi.where())
    conn = http.client.HTTPSConnection("api.reccobeats.com", context=context)
    payload = ''
    headers = {
        'Accept': 'application/json'
    }

    import signal
    import time

    mpd_dir = "./archive/data"  # e.g. "/home/user/datasets/spotify_mpd"

    def cleanup_handler(signum, frame):
        print("Handling sigkill")
        # save_cache(cache, cache_path)
        mongo_cache.close()
        exit(0)  # Exit gracefully


    signal.signal(signal.SIGTERM, cleanup_handler)

    playlists = []
    for pl in iter_mpd_playlists(mpd_dir):
        playlists.append(pl)
        if LIMIT and len(playlists) >= LIMIT:
            break

    print(f'Loaded {len(playlists)} playlists')


    track_features = augment_tracks(playlists, mongo_cache)