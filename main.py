#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
from tqdm import tqdm
import os
import json
import argparse
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# PyG hetero stuff
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

try:
    import faiss
except Exception:
    faiss = None



    # Constants
AUDIO_FEATURE_KEYS = [
    'danceability','energy','valence','tempo','loudness',
    'speechiness','instrumentalness','liveness','acousticness'
]

SEED = 42

# Cell 2: MPD iterator & small sample fallback

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

conn = http.client.HTTPSConnection("api.reccobeats.com")
payload = ''
headers = {
  'Accept': 'application/json'
}

def augment_tracks(playlists: List[Dict], cache_path: str = None, batch=50):
    """Return dict track_uri -> feature dict for AUDIO_FEATURE_KEYS. Use cache if available."""
    unique = {}
    for pl in playlists:
        for t in pl['tracks']:
            unique[t['track_uri']] = t
    tids = list(unique.keys())
    features = {}
    cache = {}
    unfound = 0
    found = 0

    def chunk_list(original_list, chunk_size=40):
        result_list = []
        # Iterate from index 0 to the end of the list, in steps of chunk_size
        for i in range(0, len(original_list), chunk_size):
            # Slice the original list and append the sublist to the result
            result_list.append(original_list[i:i + chunk_size])
        return result_list

    batch_tids = chunk_list(tids)

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    def af_to_vec(af):
        if af is None:
            return {k: 0.0 for k in AUDIO_FEATURE_KEYS}
        return {k: float(af.get(k, 0.0)) for k in AUDIO_FEATURE_KEYS}

    for one_chunk_tids in batch_tids:
        try:
            # conn.request("GET", f'/v1/audio-features?ids={track_id}', payload, headers)
            url = '/v1/audio-features?ids='
            for i, tid in enumerate(one_chunk_tids):
                track_id = tid.split(":")[-1]  # take the last part after ":"

                if track_id in cache:
                    features[track_id] = cache[track_id]
                    print("kesirano")
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
                        track_id = data_dict["href"].split("/")[-1]
                        print(track_id)
                        af = data_dict
                        vec = af_to_vec(af)
                        features[track_id] = vec
                        cache[track_id] = vec

                found = len(data_arr)
                print(f"Founds: {found}")
                print(f"Unfounds: {40 - found}")

        except Exception as e:
            print('API error:', e)
            if cache_path:
                try:
                    with open(cache_path, 'w', encoding='utf-8') as fh:
                        json.dump(cache, fh)
                except Exception as e:
                    print('Warning: failed to write cache:', e)

    if cache_path:
        try:
            with open(cache_path, 'w', encoding='utf-8') as fh:
                json.dump(cache, fh)
        except Exception as e:
            print('Warning: failed to write cache:', e)
    return features

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

def get_spotify_genres(track_spotify_id):
    all_genres = set()
    try:
        tr = sp.track(track_spotify_id)
    except:
        return all_genres

    for artist in tr["artists"]:
        try:
            art_obj = sp.artist(artist["id"])
        except:
            continue
        for g in art_obj.get("genres", []):
            all_genres.add(g.lower().replace(" ", "_"))

    return all_genres

# Cell 4: build full heterogeneous KG (NetworkX + PyG HeteroData)

def build_full_kg(playlists: List[Dict], track_features: Dict[str, Dict]=None):
    nxg = nx.MultiDiGraph()
    node_ids = {nt: {} for nt in ['playlist','track','artist','album']}
    counters = {nt: 0 for nt in node_ids}

    # Build NX graph and maps
    for pl in playlists:
        pid = pl.get('pid')
        if pid not in node_ids['playlist']:
            node_ids['playlist'][pid] = counters['playlist']
            counters['playlist'] += 1
            nxg.add_node(pid, type='playlist')

        for t in pl['tracks']:
            tid = t.get('track_uri')
            print(tid)
            art = t.get('artist_uri', 'artist:unknown')
            alb = t.get('album_uri', 'album:unknown')
            
            if tid not in node_ids['track']:
                node_ids['track'][tid] = counters['track']; counters['track'] += 1
                nxg.add_node(tid, type='track')
            if art not in node_ids['artist']:
                node_ids['artist'][art] = counters['artist']; counters['artist'] += 1
                nxg.add_node(art, type='artist')
            if alb not in node_ids['album']:
                node_ids['album'][alb] = counters['album']; counters['album'] += 1
                nxg.add_node(alb, type='album')
            nxg.add_edge(pid, tid, relation='contains')
            nxg.add_edge(tid, art, relation='by')
            nxg.add_edge(tid, alb, relation='on')

    # # Build HeteroData
    data = HeteroData()
    for ntype in ['playlist','track','artist','album']:
        ncount = counters[ntype]
        if ncount == 0:
            data[ntype].x = torch.zeros((0, len(AUDIO_FEATURE_KEYS)))
            continue
        if ntype == 'track':
            track_list = [None] * ncount
            feat_list = [None] * ncount
            for tid, idx in node_ids['track'].items():
                track_list[idx] = tid
                tf = track_features.get(tid, {k: 0.0 for k in AUDIO_FEATURE_KEYS})
                feat_list[idx] = np.array([tf.get(k, 0.0) for k in AUDIO_FEATURE_KEYS], dtype=float)
            feats_np = np.vstack(feat_list)
            feats_np = StandardScaler().fit_transform(feats_np)
            data['track'].x = torch.tensor(feats_np, dtype=torch.float)
            data['track'].tid_list = track_list
        else:
            inverse = [None] * counters[ntype]
            for k, v in node_ids[ntype].items():
                inverse[v] = k
            feat_list = []
            for node in inverse:
                deg = nxg.degree(node)
                feat_list.append([float(deg)])
            feats_np = np.vstack(feat_list) if len(feat_list)>0 else np.zeros((0,1))
            if feats_np.shape[0] > 0:
                feats_np = StandardScaler().fit_transform(feats_np)
                data[ntype].x = torch.tensor(feats_np, dtype=torch.float)
            else:
                data[ntype].x = torch.zeros((0,1), dtype=torch.float)
            data[ntype].id_list = inverse
    #
    # # edges: playlist->track, track->artist, track->album with reverse relations
    # def edges_from_nx(source_type, target_type, relation):
    #     src_idxs = []
    #     dst_idxs = []
    #     for u, v, d in nxg.edges(data=True):
    #         if d.get('relation') != relation:
    #             continue
    #         if nxg.nodes[u].get('type') != source_type or nxg.nodes[v].get('type') != target_type:
    #             continue
    #         src_idxs.append(node_ids[source_type][u])
    #         dst_idxs.append(node_ids[target_type][v])
    #     if len(src_idxs) == 0:
    #         return torch.empty((2,0), dtype=torch.long)
    #     return torch.tensor([src_idxs, dst_idxs], dtype=torch.long)
    #
    # data['playlist', 'contains', 'track'].edge_index = edges_from_nx('playlist','track','contains')
    # if data['playlist', 'contains', 'track'].edge_index.numel() > 0:
    #     data['track', 'rev_contains', 'playlist'].edge_index = data['playlist', 'contains', 'track'].edge_index.flip(0)
    # else:
    #     data['track', 'rev_contains', 'playlist'].edge_index = torch.empty((2,0), dtype=torch.long)
    #
    # data['track', 'by', 'artist'].edge_index = edges_from_nx('track','artist','by')
    # if data['track','by','artist'].edge_index.numel() > 0:
    #     data['artist', 'rev_by', 'track'].edge_index = data['track','by','artist'].edge_index.flip(0)
    # else:
    #     data['artist', 'rev_by', 'track'].edge_index = torch.empty((2,0), dtype=torch.long)
    #
    # data['track', 'on', 'album'].edge_index = edges_from_nx('track','album','on')
    # if data['track','on','album'].edge_index.numel() > 0:
    #     data['album', 'rev_on', 'track'].edge_index = data['track','on','album'].edge_index.flip(0)
    # else:
    #     data['album', 'rev_on', 'track'].edge_index = torch.empty((2,0), dtype=torch.long)
    #
    return nxg, node_ids


# In[17]:


mpd_dir = "./archive/data"  # e.g. "/home/user/datasets/spotify_mpd"
use_spotify = True
spotify_client_id = None
spotify_client_secret = None
cache_path = 'track_features_cache.json'
sample_rate = 0.05
limit = 2000
device = 'cpu'
epochs = 8

playlists = []
for pl in iter_mpd_playlists(mpd_dir, 1):
    playlists.append(pl)
    if limit and len(playlists) >= limit:
        break

print(f'Loaded {len(playlists)} playlists')


# In[18]:


if not os.path.exists(cache_path):
    with open(cache_path, 'w') as file:
        file.write("")
    print(f"File '{cache_path}' created successfully.")
else:
    print(f"File '{cache_path}' already exists. No new file was created.")
track_features = augment_tracks(playlists, cache_path=cache_path)


# nxg, node_id_maps = build_full_kg(playlists)
# print('HeteroData node types:', hetero_data.node_types, 'edge types:', hetero_data.edge_types)
# print(nxg)


# In[13]:


print(track_features)


# In[ ]:




