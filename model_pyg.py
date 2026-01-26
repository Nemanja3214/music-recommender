import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv
import torch_geometric.transforms as T

from torch_geometric.data import HeteroData
from build_graph_network import SpotifyHeteroGraphBuilder
from collections import defaultdict
import matplotlib.pyplot as plt

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
# Hetero GNN model (build convs from given relation list)
# -------------------------
class HeteroGNN(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # Build conv modules only for relations present in edge_types
        # conv1: input -> hidden
        self.conv1 = HeteroConv(
            {
                (src, rel, dst): SAGEConv(in_channels_dict[src], hidden_channels)
                for (src, rel, dst) in self.edge_types
            },
            aggr='mean'
        )

        # conv2: hidden -> out
        self.conv2 = HeteroConv(
            {
                (src, rel, dst): SAGEConv(hidden_channels, out_channels)
                for (src, rel, dst) in self.edge_types
            },
            aggr='mean'
        )
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in metadata[0]})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(out_channels) for nt in metadata[0]})

        self.res_proj = nn.ModuleDict({
            ntype: nn.Linear(hidden_channels, out_channels) for ntype in node_types
        })

        self.post_lin = nn.ModuleDict()
        for ntype in self.node_types:
            self.post_lin[ntype] = nn.Linear(out_channels, out_channels)
        print(self)

    def forward(self, x_dict, edge_index_dict):
        # ----- Conv1: input -> hidden -----
        x1 = self.conv1(x_dict, edge_index_dict)
        x1 = {
            k: F.dropout(F.relu(self.norm1[k](v)), p=self.dropout, training=self.training)
            for k, v in x1.items()
        }

        # ----- Conv2: hidden -> out with residual -----
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {
            k: self.norm2[k](v + self.res_proj[k](x1[k]))
            for k, v in x2.items()
        }

        # ----- Optional final linear projection -----
        out = {k: self.post_lin[k](v) for k, v in x2.items()}

        return out

# ------------------------------
# SIMPLE PLUG-IN DIAGNOSTIC FUNCTION
# ------------------------------
def diagnose_batch(model, batch, target_edge, device):
    import torch
    import torch.nn.functional as F

    batch = batch.to(device)
    model.zero_grad()

    out = model(batch.x_dict, batch.edge_index_dict)

    p_idx, t_idx = batch[target_edge].edge_label_index
    labels = batch[target_edge].edge_label

    z_p = out['Playlist'][p_idx]
    z_t = out['Track'][t_idx]
    scores = (z_p * z_t).sum(dim=1)

    # group by playlist
    from collections import defaultdict
    pos_by_p = defaultdict(list)
    neg_by_p = defaultdict(list)

    for i in range(len(scores)):
        pid = p_idx[i].item()
        if labels[i] == 1:
            pos_by_p[pid].append(scores[i])
        else:
            neg_by_p[pid].append(scores[i])

    pos_scores = []
    neg_scores = []
    for pid in pos_by_p:
        if pid not in neg_by_p:
            continue
        for ps in pos_by_p[pid]:
            ns = neg_by_p[pid][torch.randint(0, len(neg_by_p[pid]), (1,))]
            pos_scores.append(ps)
            neg_scores.append(ns)

    if len(pos_scores) == 0:
        print("⚠ No valid pos-neg pairs in batch")
        return

    pos_scores = torch.stack(pos_scores)
    neg_scores = torch.stack(neg_scores)

    loss = F.softplus(-(pos_scores - neg_scores)).mean()
    loss.backward()

    print("\n================ DIAGNOSTICS ================\n")

    print("=== EMBEDDINGS ===")
    print(f"Playlist std={z_p.std():.4f}, mean={z_p.mean():.4f}")
    print(f"Track    std={z_t.std():.4f}, mean={z_t.mean():.4f}")

    print("\n=== SCORES ===")
    print(f"BPR loss: {loss.item():.4f}")
    print(f"Pos mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}")
    print(f"Neg mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}")
    print(f"Margin (pos-neg): {(pos_scores.mean() - neg_scores.mean()):.4f}")

    print("\n=== SCORE HEALTH ===")
    print(f"Frac(pos > neg): {(pos_scores > neg_scores).float().mean():.3f}")
    print(f"Frac(margin > 1): {((pos_scores - neg_scores) > 1).float().mean():.3f}")

    print("\n=== GRADIENT FLOW ===")
    dead = []
    tiny = []

    for name, p in model.named_parameters():
        if p.grad is None:
            dead.append(name)
        else:
            g = p.grad
            maxg = g.abs().max().item()
            print(f"{name}: std={g.std():.2e}, max={maxg:.2e}")
            if maxg < 1e-7:
                tiny.append(name)

    print(f"\nDead tensors: {len(dead)}")
    if dead:
        print("Dead layers:")
        for d in dead:
            print(" ", d)

    if tiny:
        print("\nNear-zero gradient layers:")
        for t in tiny:
            print(" ", t)

    print("\n=============================================\n")




# -------------------------
# Main usage
# -------------------------
if __name__ == "__main__":
    builder = SpotifyHeteroGraphBuilder()
    data, nodes_id_map = builder.run()

    pt = ("Playlist", "Has Track", "Track")
    rev_pt = ("Track", "rev_Has Track", "Playlist")

    # print_pyg_diagnostics(data)

    transform = T.Compose([T.ToUndirected()])
    data = transform(data)
    print_pyg_diagnostics(data)

    split = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.1,
        is_undirected=True,
        key="edge_label",
        add_negative_train_samples=True,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=20,
        edge_types=[pt],
        rev_edge_types=[rev_pt]
    )
    train_data, val_data, test_data = split(data)
    print("TRAIN")
    print(train_data)

    print("TEST")
    print(test_data)
    # Build model from final metadata
    in_dims = {ntype: train_data[ntype].x.size(1) for ntype in data.node_types}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(metadata=train_data.metadata(),
                      in_channels_dict=in_dims,
                      hidden_channels=256,
                      out_channels=128,
                      dropout=0.2)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    # Move all data to device
    for ntype in train_data.node_types:
        if train_data[ntype].x is not None:
            train_data[ntype].x = train_data[ntype].x.to(device)
    for etype in train_data.edge_types:
        if train_data[etype].edge_index is not None:
            train_data[etype].edge_index = train_data[etype].edge_index.to(device)

    # Node counts for negative sampling
    node_counts = {ntype: train_data[ntype].num_nodes for ntype in train_data.node_types}

    batch_size = 256
    epochs = 15
    num_neighbors = [50, 25]
    seed_node_type = None
    target_edge = "Playlist" ,"Has Track", "Track"

    train_edge_label_index = train_data[target_edge].edge_label_index
    train_edge_label = train_data[target_edge].edge_label
    print(f"LABELS {train_edge_label}")

    # val_edge_label_index = val_data[target_edge].edge_label_index
    # val_edge_label = val_data[target_edge].edge_label

    test_edge_label_index = test_data[target_edge].edge_label_index
    test_edge_label = test_data[target_edge].edge_label


    def check_train_test_overlap(train_data, test_data, edge_type=('Playlist', 'Has Track', 'Track')):
        # Train positives
        train_mask = train_data[edge_type].edge_label == 1
        train_edges = set(tuple(x.tolist()) for x in train_data[edge_type].edge_label_index[:, train_mask].t())

        # Test positives
        test_mask = test_data[edge_type].edge_label == 1
        test_edges = set(tuple(x.tolist()) for x in test_data[edge_type].edge_label_index[:, test_mask].t())

        # Include reversed edges
        rev_train_edges = set((dst, src) for src, dst in train_edges)
        rev_test_edges = set((dst, src) for src, dst in test_edges)

        all_train_edges = train_edges.union(rev_train_edges)
        all_test_edges = test_edges.union(rev_test_edges)

        overlap = all_train_edges.intersection(all_test_edges)
        print("Number of overlapping POSITIVE edges (including reversed):", len(overlap))


    check_train_test_overlap(train_data, test_data)

    trainloader = LinkNeighborLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        edge_label_index=(target_edge, train_edge_label_index),
        edge_label=train_edge_label,
        num_neighbors=num_neighbors,
    )
    # valloader = LinkNeighborLoader(
    #     val_data,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     edge_label_index=(target_edge, val_edge_label_index),
    #     edge_label=val_edge_label,
    #     num_neighbors=num_neighbors,
    # )
    testloader = LinkNeighborLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        edge_label_index=(target_edge, test_edge_label_index),
        edge_label=test_edge_label,
        num_neighbors=num_neighbors,
    )
    criterion = nn.TripletMarginLoss(margin=1.0)

    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        total_pos_epoch = 0
        total_neg_epoch = 0

        batch_count = 0
        skipped_count = 0

        for batch in trainloader:
            # diagnose_batch(model, batch, target_edge, device)
            # exit()
            optimizer.zero_grad()
            batch_count += 1
            selected_batch = batch[target_edge]

            out = model(batch.x_dict, batch.edge_index_dict)

            edge_type = ('Playlist', 'Has Track', 'Track')

            # get ids
            p_idx, t_idx = batch[edge_type].edge_label_index
            # get 1s and 0s, same size as p_idx and t_idx
            labels = batch[edge_type].edge_label

            # get corresponding playlists and tracks embeddings
            z_p = out['Playlist'][p_idx]
            z_t = out['Track'][t_idx]
            scores = (z_p * z_t).sum(dim=1)

            pos_by_playlist = defaultdict(list)
            neg_by_playlist = defaultdict(list)

            # separate positives and negatives by pidx
            for i in range(len(scores)):
                # get playlist id from pidx
                pid = p_idx[i].item()
                if labels[i] == 1:
                    pos_by_playlist[pid].append(scores[i])
                else:
                    neg_by_playlist[pid].append(scores[i])


            pos_scores = []
            neg_scores = []

            for pid in pos_by_playlist:
                # add only scores that have negative
                if pid not in neg_by_playlist:
                    continue

                for ps in pos_by_playlist[pid]:
                    # add random negative link for that playlist
                    ns = neg_by_playlist[pid][
                        torch.randint(0, len(neg_by_playlist[pid]), (1,))
                    ]
                    pos_scores.append(ps)
                    neg_scores.append(ns)


            if len(pos_scores) == 0:
                skipped_count += 1
                continue


            pos_scores = torch.stack(pos_scores)
            neg_scores = torch.stack(neg_scores)

            # BPR loss
            loss = F.softplus(-(pos_scores - neg_scores)).mean()

            loss.backward()
            optimizer.step()
            # print(f"Batch loss {float(loss)}")
            total_loss_epoch += float(loss)

        print(f"Epoch {epoch:03d} | Loss: {total_loss_epoch:.4f}"
              f"Skipped {skipped_count}/{batch_count}")
        epoch_losses.append(total_loss_epoch)

    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.savefig("Loss.png")

    print(f"STD track: {out['Track'].std(dim=0).mean()}")
    print(f"STD playlist: {out['Playlist'].std(dim=0).mean()}")

    model.eval()
    train_edge_type = ('Playlist', 'Has Track', 'Track')
    p_idx_train, t_idx_train = train_data[train_edge_type].edge_label_index
    labels_train = train_data[train_edge_type].edge_label

    train_tracks_by_playlist = defaultdict(set)
    for pid, tid, lbl in zip(p_idx_train.tolist(), t_idx_train.tolist(), labels_train.tolist()):
        if lbl == 1:
            train_tracks_by_playlist[pid].add(tid)

    with torch.no_grad():

        out = model(train_data.x_dict, train_data.edge_index_dict)

        playlist_emb = out['Playlist']

        # Track embeddings (only those in training set)
        track_emb = out["Track"]
        track_ids = torch.arange(track_emb.size(0), device=device)

        test_edge_type = ('Playlist', 'Has Track', 'Track')
        p_idx, t_idx = test_data[test_edge_type].edge_label_index
        labels = test_data[test_edge_type].edge_label

        # Only keep positive edges (label == 1)
        pos_mask = labels == 1
        p_idx_pos = p_idx[pos_mask]
        t_idx_pos = t_idx[pos_mask]

        # Group ground-truth tracks by playlist
        gt_tracks_by_playlist = defaultdict(list)
        for pid, tid in zip(p_idx_pos.tolist(), t_idx_pos.tolist()):
            gt_tracks_by_playlist[pid].append(tid)

        K = 20
        hit_count = 0
        recall_sum = 0
        num_playlists = len(gt_tracks_by_playlist)

        for pid, gt_tids in gt_tracks_by_playlist.items():
            p_vec = playlist_emb[pid].unsqueeze(0)  # [1, embed_dim]

            # similarity with training tracks
            sims = torch.matmul(p_vec, track_emb.t()).squeeze(0)  # [num_train_tracks]

            ignore_tids = train_tracks_by_playlist.get(pid, set())
            if ignore_tids:
                mask = torch.ones_like(sims, dtype=torch.bool)
                ignore_indices = torch.tensor(list(ignore_tids), device=sims.device)
                mask[ignore_indices] = 0
                sims_masked = sims.clone()
                sims_masked[~mask] = -float('inf')  # ensure ignored tracks don't appear in topk
            else:
                sims_masked = sims

            # top-K recommended track indices (relative to train tracks)
            topk_vals, topk_index = torch.topk(sims_masked, K)
            topk_track_ids = track_ids[topk_index].tolist()

            # Hit Rate@K: at least one ground-truth track in top-K
            if any(tid in topk_track_ids for tid in gt_tids):
                hit_count += 1

            # Recall@K: fraction of ground-truth tracks in top-K
            recall_sum += sum(tid in topk_track_ids for tid in gt_tids) / len(gt_tids)

        hit_rate = hit_count / num_playlists
        recall_at_k = recall_sum / num_playlists

        print(f"Hit Rate@{K}: {hit_rate:.4f}")
        print(f"Recall@{K}: {recall_at_k:.4f}")
