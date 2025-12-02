#!/usr/bin/env python
# coding: utf-8

import os
import json


from mongo import MongoCache
from music_graph import SpotifyMusicGraphSchema

try:
    import faiss
except Exception:
    faiss = None

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

def build_full_kg(playlists, cache: MongoCache, graph, track_features = None,):
    # iterate playlists
    track_counter = 0
    for pl in playlists:
        for t in pl.get("tracks", []):
            tid = t["track_uri"]
            # add track features
            # tr_features = track_features[tid]
            # print(t)
            track_counter += 1
            cached = cache.get(tid, cache.ARTISTS_NAME)
            if cached is None:
                print(f"\n\n there is {track_counter} tarcks")
                return graph
            artist_entries = cached.get("artists", [])

            for a in artist_entries:
                art_uri = a.get("uri")
                # print(art_uri)
                if not art_uri:
                    continue
                graph.add_artist(art_uri, a.get("name"), a.get("genres", []), {
                    "num_followers": a["followers"]["total"],
                    "popularity": a["popularity"]
                })

            # Album
            album_uri = t.get("album_uri")
            # print(album_uri, t.get("album_name"), [a.get("uri") for a in artist_entries])
            if album_uri:
                graph.add_album(album_uri, t.get("album_name"), [a.get("uri") for a in artist_entries])

            track_features = cache.get(tid, cache.TRACKS_NAME)
            if track_features is None:
                continue
            print(tid)
            track_features["duration"] = t["duration_ms"]
            track_features["pos"] = t["pos"]
            del track_features["_id"]
            graph.add_track(tid, t["track_name"], album_uri, [a.get("uri") for a in artist_entries], track_features)
            pid = pl["pid"]  # assume present
            graph.add_playlist(graph.SPOTIFY_PLAYLIST_PREFIX + str(pid), [t["track_uri"] for track in pl.get("tracks", [])], pl["name"], {
                "modified_at": pl["modified_at"],
                "num_tracks": pl["num_tracks"],
                "num_albums": pl["num_albums"],
                "num_followers": pl["num_followers"]
            })

    return graph

limit = 50000
mpd_dir = "./archive/data"
graph = SpotifyMusicGraphSchema()
cache = MongoCache()

playlists = []
for pl in iter_mpd_playlists(mpd_dir):
    playlists.append(pl)
    if limit and len(playlists) >= limit:
        break
graph = build_full_kg(playlists, cache, graph)
graph.serialize()
# graph.visualize_rdf_graph()
# print('HeteroData node types:', hetero_data.node_types, 'edge types:', hetero_data.edge_types)
# print(nxg)





