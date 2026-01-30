import os
import random

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

def sample_global_negatives_for_playlist(
    pid: int,
    num_tracks: int,
    positives_set,
    m: int,
    device,
):
    # Sample m track ids uniformly from [0, num_tracks)
    # excluding positives. (Allows repeats avoidance via while.)
    negs = []
    tries = 0
    max_tries = max(1000, m * 50)

    while len(negs) < m and tries < max_tries:
        cand = random.randrange(num_tracks)
        if cand not in positives_set:
            negs.append(cand)
        tries += 1

    if len(negs) < m:
        # fallback: just return what we got (better than crashing mid-train)
        # but you can raise if you prefer
        pass

    return torch.tensor(negs, dtype=torch.long, device=device)


def print_playlist_track_ranking(builder, train_tracks_by_playlist, playlist_emb, track_emb, K=20):
    from collections import defaultdict
    import torch

    # Mapping from IDs to names
    playlist_names_map = {v: builder.node_names.get(k, f"Playlist_{v}")
                          for k, v in builder.nodes_id_map['Playlist'].items()}
    track_names_map = {v: builder.node_names.get(k, f"Track_{v}")
                       for k, v in builder.nodes_id_map['Track'].items()}

    print("\n===== PLAYLIST TRACK RANKINGS =====\n")

    for pid, connected_tids in train_tracks_by_playlist.items():
        p_vec = playlist_emb[pid].unsqueeze(0)  # [1, embed_dim]
        sims = torch.matmul(p_vec, track_emb.t()).squeeze(0)  # similarity with all tracks

        # --- Connected tracks ---
        connected_tracks = [track_names_map[i] for i in connected_tids]
        train_sims = sims[list(connected_tids)] if connected_tids else torch.tensor([])
        print(f"Playlist: {playlist_names_map.get(pid, f'Playlist_{pid}')} (ID {pid})")
        print(f"  Connected tracks ({len(connected_tracks)}): {connected_tracks}")
        if len(train_sims) > 0:
            print(f"  Max similarity among connected: {train_sims.max().item():.3f}")
            print(f"  Min similarity among connected: {train_sims.min().item():.3f}")

        # --- Top K including connected tracks ---
        topk_vals_all, topk_index_all = torch.topk(sims, K)
        topk_tracks_all = [track_names_map[i.item()] for i in topk_index_all]
        print(f"  Top {K} tracks (including connected): {topk_tracks_all}")

        # --- Top K excluding connected tracks ---
        if connected_tids:
            mask = torch.ones_like(sims, dtype=torch.bool)
            mask[list(connected_tids)] = 0
            sims_masked = sims.clone()
            sims_masked[~mask] = -float('inf')
        else:
            sims_masked = sims

        topk_vals_masked, topk_index_masked = torch.topk(sims_masked, K)
        topk_tracks_masked = [track_names_map[i.item()] for i in topk_index_masked]
        print(f"  Top {K} tracks (excluding connected): {topk_tracks_masked}\n")



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
                (src, rel, dst): SAGEConv(in_channels_dict[src], hidden_channels, normalize=True)
                for (src, rel, dst) in self.edge_types
            },
            aggr='mean'
        )

        # conv2: hidden -> out
        self.conv2 = HeteroConv(
            {
                (src, rel, dst): SAGEConv(hidden_channels, out_channels, normalize=True)
                for (src, rel, dst) in self.edge_types
            },
            aggr='mean'
        )

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
            k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
            for k, v in x1.items()
        }

        # ----- Conv2: hidden -> out with residual -----
        x2 = self.conv2(x1, edge_index_dict)
        x2 = {
            k: v + self.res_proj[k](x1[k])
            for k, v in x2.items()
        }

        return x2

        # ----- Optional final linear projection -----
        # out = {k: self.post_lin[k](v) for k, v in x2.items()}
        #
        # return out


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

# -------------------------
# Main usage
# -------------------------
if __name__ == "__main__":
    builder = SpotifyHeteroGraphBuilder()
    data, nodes_id_map = builder.run()

    pt = ("Playlist", "Has Track", "Track")
    rev_pt = ("Track", "rev_Has Track", "Playlist")

    # print_pyg_diagnostics(data)

    transform = T.Compose([T.NormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomLinkSplit(
                                num_val=0.1,
                                num_test=0.2,
                                is_undirected=True,
                                add_negative_train_samples=False,
                                disjoint_train_ratio=0.0,
                                neg_sampling_ratio=5,
                                edge_types=[pt],
                                rev_edge_types=[rev_pt]
                            )
                           ])
    train_data, val_data, test_data = transform(data)
    print_pyg_diagnostics(data)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    # Move all data to device
    for ntype in train_data.node_types:
        if train_data[ntype].x is not None:
            train_data[ntype].x = train_data[ntype].x.to(device)
    for etype in train_data.edge_types:
        if train_data[etype].edge_index is not None:
            train_data[etype].edge_index = train_data[etype].edge_index.to(device)

    # Node counts for negative sampling
    node_counts = {ntype: train_data[ntype].num_nodes for ntype in train_data.node_types}

    batch_size = 1024
    epochs = 10
    num_neighbors = [15, 10]
    seed_node_type = None
    target_edge = "Playlist" ,"Has Track", "Track"

    train_edge_label_index = train_data[target_edge].edge_label_index
    train_edge_label = train_data[target_edge].edge_label
    print(f"LABELS {train_edge_label}")

    # val_edge_label_index = val_data[target_edge].edge_label_index
    # val_edge_label = val_data[target_edge].edge_label

    test_edge_label_index = test_data[target_edge].edge_label_index
    test_edge_label = test_data[target_edge].edge_label

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

    epoch_losses = []

    p_idx_train, t_idx_train = train_data[pt].edge_label_index
    labels_train = train_data[pt].edge_label

    train_tracks_by_playlist = defaultdict(set)
    for pid, tid, lbl in zip(p_idx_train.tolist(), t_idx_train.tolist(), labels_train.tolist()):
        if lbl == 1:
            train_tracks_by_playlist[pid].add(tid)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        total_pos_epoch = 0
        total_neg_epoch = 0

        batch_count = 0
        skipped_count = 0

        used_batches = 0
        pairs_used = 0

        pos_mean_sum = 0.0
        neg_mean_sum = 0.0
        margin_mean_sum = 0.0

        # optional: for crude AUC on sampled pairs
        auc_correct = 0
        auc_total = 0

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
            z_p = F.normalize(z_p, dim=1)
            z_t = F.normalize(z_t, dim=1)

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
                if pid not in neg_by_playlist:
                    continue

                # stack negatives for this playlist
                negs = torch.stack(neg_by_playlist[pid])  # shape: [num_neg]

                # take hardest negative (highest score)
                hardest_neg = negs.max()

                for ps in pos_by_playlist[pid]:
                    pos_scores.append(ps)
                    neg_scores.append(hardest_neg)


            if len(pos_scores) == 0:
                skipped_count += 1
                continue


            pos_scores = torch.stack(pos_scores)
            neg_scores = torch.stack(neg_scores)

            used_batches += 1
            pairs_used += pos_scores.numel()

            pos_m = pos_scores.mean().item()
            neg_m = neg_scores.mean().item()
            margin_m = (pos_scores - neg_scores).mean().item()

            pos_mean_sum += pos_m
            neg_mean_sum += neg_m
            margin_mean_sum += margin_m

            # optional AUC over sampled pairs: fraction where pos > neg
            auc_correct += (pos_scores > neg_scores).sum().item()
            auc_total += pos_scores.numel()

            loss = F.softplus(-(pos_scores - neg_scores)).mean()  # stable BPR)

            loss.backward()
            optimizer.step()
            # print(f"Batch loss {float(loss)}")
            total_loss_epoch += float(loss)

        avg_loss_used = total_loss_epoch / max(1, used_batches)
        avg_pos = pos_mean_sum / max(1, used_batches)
        avg_neg = neg_mean_sum / max(1, used_batches)
        avg_margin = margin_mean_sum / max(1, used_batches)
        auc = auc_correct / max(1, auc_total)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss(sum): {total_loss_epoch:.4f} | Loss/used_batch: {avg_loss_used:.4f} | "
            f"pairs: {pairs_used} | "
            f"pos_mean: {avg_pos:.4f} | neg_mean: {avg_neg:.4f} | margin: {avg_margin:.4f} | "
            f"AUC~: {auc:.4f} | "
            f"Skipped {skipped_count}/{batch_count}"
        )

        model.eval()
        with torch.no_grad():
            out = model(train_data.x_dict, train_data.edge_index_dict)
            print("Epoch", epoch,
                  "| Std Track:", out['Track'].std(dim=0).mean().item(),
                  "| Std Playlist:", out['Playlist'].std(dim=0).mean().item())

        model.train()
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

        playlist_emb = F.normalize(out['Playlist'], dim=1)
        track_emb = F.normalize(out['Track'], dim=1)

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

        K = 100
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
        print_playlist_track_ranking(builder, train_tracks_by_playlist, playlist_emb, track_emb, K=20)
