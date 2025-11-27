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
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# PyG hetero stuff
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

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
    for pl in playlists:
        for t in pl.get("tracks", []):
            tid = t["track_uri"]
            # add track features
            # tr_features = track_features[tid]
            print(t)

            cached = cache.get(tid, cache.ARTISTS_NAME)
            artist_entries = cached.get("artists", [])

            for a in artist_entries:
                art_uri = a.get("uri")
                if not art_uri:
                    continue

                graph.add_artist(art_uri, a.get("name"), a.get("genres", []))

            # Album
            album_uri = t.get("album_uri")
            if album_uri:
                graph.add_album(album_uri, t.get("album_name"), [a.get("uri") for a in artist_entries])

            graph.add_track(tid, t["track_name"], album_uri, [a.get("uri") for a in artist_entries])
            pid = pl["pid"]  # assume present
            graph.add_playlist(graph.SPOTIFY_PLAYLIST_PREFIX + str(pid), [t["track_uri"] for track in pl.get("tracks", [])])

    return graph

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


limit = 10
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
graph.visualize_rdf_graph()
# print('HeteroData node types:', hetero_data.node_types, 'edge types:', hetero_data.edge_types)
# print(nxg)





