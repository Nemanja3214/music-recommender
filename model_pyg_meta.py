import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv, MetaPath2Vec
from torch_geometric.data import HeteroData
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling

from torch_geometric.data import HeteroData
from build_graph_network import SpotifyHeteroGraphBuilder

import torch_sparse
print(torch_sparse.__version__)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------
# Diagnostics / utils
# -------------------------
def print_pyg_diagnostics(data: HeteroData):
    print("\n================ HETERO DATA DIAGNOSTICS ================\n")
    print("Node types:", data.node_types)
    print("Edge types:", data.edge_types, "\n")

    print("---- NODE FEATURES ----")
    for ntype in data.node_types:
        x = data[ntype].get("x", None)
        if x is None:
            print(f" ❌ {ntype}: x = None")
        else:
            print(f" ✔ {ntype}: x.shape = {tuple(x.shape)}, dtype={x.dtype}")
    print()

    print("---- EDGE INDEXES ----")
    for etype in data.edge_types:
        ei = data[etype].get("edge_index", None)
        if ei is None:
            print(f" ❌ {etype}: edge_index = None")
        else:
            print(f" ✔ {etype}: edge_index.shape = {tuple(ei.shape)}, dtype={ei.dtype}")
    print()

    print("---- EMPTY RELATIONS ----")
    for etype in data.edge_types:
        ei = data[etype].get("edge_index", None)
        if ei is not None and ei.size(1) == 0:
            print(f" ⚠ EMPTY EDGE TYPE: {etype}")
    print()

    print("---- CHECK FOR ISOLATED NODE TYPES ----")
    connected = set()
    for (src, rel, dst) in data.edge_types:
        ei = data[(src, rel, dst)].edge_index
        if ei is not None and ei.size(1) > 0:
            connected.add(src)
            connected.add(dst)
    for ntype in data.node_types:
        if ntype not in connected:
            print(f" ⚠ WARNING: Node type '{ntype}' has NO connected edges!")
    print()

    print("---- PER-RELATION INPUT DIM CHECK ----")
    for (src, rel, dst) in data.edge_types:
        src_x = data[src].get("x", None)
        dst_x = data[dst].get("x", None)
        print(f"Relation {src}-{rel}->{dst}: "
              f"src_dim={None if src_x is None else src_x.size(1)}, "
              f"dst_dim={None if dst_x is None else dst_x.size(1)}")
    print("\n================ END DIAGNOSTICS ================\n")


# -------------------------
# Main usage
# -------------------------
if __name__ == "__main__":

    metapath = [
        ('Track', 'rev_Has Track', 'Playlist'),
        ('Playlist', 'Has Track', 'Track'),
        ('Track', 'byArtist', 'Artist'),
        ('Artist', 'rev_byArtist', 'Track'),
    ]

    builder = SpotifyHeteroGraphBuilder()
    data, nodes_id_map = builder.run()

    transform = T.Compose([T.ToUndirected()])
    data = transform(data)
    print_pyg_diagnostics(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                         metapath=metapath, walk_length=5, context_size=5,
                         walks_per_node=1, num_negative_samples=5,
                         sparse=True).to(device)

    batch_size = 128
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    # Node counts for negative sampling
    node_counts = {ntype: data[ntype].num_nodes for ntype in data.node_types}


    device = 'cpu'
    epochs = 50
    target_edge = "Playlist" ,"Has Track", "Track"


    def train(epoch, log_steps=50):
        model.train()
        total_loss = 0

        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print(
                    f'Epoch {epoch}, '
                    f'Step {i + 1}/{len(loader)}, '
                    f'Loss {total_loss / log_steps:.4f}'
                )
                total_loss = 0

    @torch.no_grad()
    def test_next_track_recommendation(model, data, train_ratio=0.8, top_k=5):
        model.eval()

        # Get all playlist-track edges
        edge_index = data[('Playlist', 'Has Track', 'Track')].edge_index  # [2, num_edges]
        num_edges = edge_index.size(1)

        # Random train/test split of edges
        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[:int(num_edges * train_ratio)]]
        test_edges = edge_index[:, perm[int(num_edges * train_ratio):]]

        # Build mapping: playlist_id -> tracks already in playlist (train)
        from collections import defaultdict
        playlist_tracks_train = defaultdict(set)
        for p, t in zip(train_edges[0].tolist(), train_edges[1].tolist()):
            playlist_tracks_train[p].add(t)

        # Get embeddings
        playlist_emb = model('Playlist')
        track_emb = model('Track')

        recall_sum = 0.0
        hit_sum = 0.0
        mrr_sum = 0.0

        test_playlists = test_edges[0].unique()

        for p in test_playlists.tolist():
            # Tracks already in the playlist (train)
            existing_tracks = playlist_tracks_train[p]

            # Candidate scores 10x128 at the end
            scores = torch.matmul(track_emb, playlist_emb[p])

            # Mask out tracks already in playlist
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[list(existing_tracks)] = 0
            scores = scores * mask.float()
            # print(existing_tracks)
            # print(mask)
            # Top-K recommendations
            top_values, top_indices = torch.topk(scores, top_k)
            recommended = top_indices.tolist()

            # Ground-truth next tracks for this playlist (from test_edges)
            gt_mask = (test_edges[0] == p)
            gt_tracks = test_edges[1, gt_mask].tolist()

            # Recall@K
            hits = len(set(recommended) & set(gt_tracks))
            recall = hits / len(gt_tracks)
            recall_sum += recall

            # Hit@K
            hit = 1.0 if hits > 0 else 0.0
            hit_sum += hit

            # MRR
            rr = 0.0
            for rank, track in enumerate(recommended, start=1):
                if track in gt_tracks:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

        n = len(test_playlists)
        print(f"Recall@{top_k}: {recall_sum / n:.4f}")
        print(f"Hit@{top_k}: {hit_sum / n:.4f}")
        print(f"MRR: {mrr_sum / n:.4f}")


    for epoch in range(1, 1000):
        train(epoch)
        model.eval()
        test_next_track_recommendation(model, data, train_ratio=0.8, top_k=5)
        print(f'Epoch: {epoch}')