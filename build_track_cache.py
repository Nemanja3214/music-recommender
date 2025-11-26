import json
import os
from typing import Dict, List

from mongo import MongoCache


# Constants


# Cell 2: MPD iterator & small sample fallback

def spotify_url_to_uri(url: str) -> str:
    # Example: https://open.spotify.com/track/6GIrIt2M39wEGwjCQjGChX
    parts = url.rstrip("/").split("/")
    item_type = parts[-2]
    item_id = parts[-1].split("?")[0]
    return f"spotify:{item_type}:{item_id}"

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

import http.client

def augment_tracks(playlists: List[Dict], cache: MongoCache, batch_size=40):
    """Return dict track_uri -> feature dict for AUDIO_FEATURE_KEYS. Use cache if available."""
    unique = {}
    for pl in playlists:
        for t in pl['tracks']:
            unique[t['track_uri']] = t
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
                break
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

    conn = http.client.HTTPSConnection("api.reccobeats.com")
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
    limit = 50000
    for pl in iter_mpd_playlists(mpd_dir):
        playlists.append(pl)
        if limit and len(playlists) >= limit:
            break

    print(f'Loaded {len(playlists)} playlists')


    track_features = augment_tracks(playlists, mongo_cache)