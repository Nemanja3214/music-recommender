import os
import random
import math
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


def sample_bpr_negatives_train_only(p_pos, num_tracks, train_pos_set, num_neg=10):
    """
    Train-only negative sampling:
    For each playlist p_pos[i], sample num_neg tracks that are NOT connected to p in TRAIN.
    Returns t_neg: [E, num_neg]
    """
    device = p_pos.device
    E = p_pos.size(0)

    t_neg = torch.randint(0, num_tracks, (E, num_neg), device=device)

    for i in range(E):
        p = int(p_pos[i].item())
        for j in range(num_neg):
            tn = int(t_neg[i, j].item())
            while (p, tn) in train_pos_set:
                tn = random.randrange(num_tracks)
            t_neg[i, j] = tn

    return t_neg

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
    def __init__(self, metadata, in_channels_dict, hidden_channels=256, out_channels=128, dropout=0.2, num_nodes_dict=None):
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # Learnable ID embeddings (big win for Playlist)
        self.id_emb = nn.ModuleDict()
        self.use_id_emb = {"Playlist", "Artist"}  # instead of adding all ntypes

        if num_nodes_dict is not None:
            for ntype in node_types:
                self.id_emb[ntype] = nn.Embedding(num_nodes_dict[ntype], in_channels_dict[ntype])
                # self.use_id_emb.add(ntype)

        # Build conv modules only for relations present in edge_types
        # conv1: input -> hidden
        self.conv1 = HeteroConv(
            {
                (src, rel, dst): SAGEConv(in_channels_dict[src], hidden_channels, normalize=False)
                for (src, rel, dst) in self.edge_types
            },
            aggr='mean'
        )

        # conv2: hidden -> out
        self.conv2 = HeteroConv(
            {
                (src, rel, dst): SAGEConv(hidden_channels, out_channels, normalize=False)
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

        x_dict = dict(x_dict)
        for ntype in x_dict:
            if ntype in self.use_id_emb:
                # assumes nodes are 0..N-1
                ids = torch.arange(x_dict[ntype].size(0), device=x_dict[ntype].device)
                x_dict[ntype] = x_dict[ntype] + self.id_emb[ntype](ids)

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
    if len(overlap) > 0:
        exit()

# -------------------------
# Main usage
# -------------------------
if __name__ == "__main__":
    builder = SpotifyHeteroGraphBuilder()
    data, nodes_id_map = builder.run()


    edge_type = ("Playlist", "Has Track", "Track")
    rev_edge_type = ("Track", "rev_Has Track", "Playlist")

    # print_pyg_diagnostics(data)

    transform = T.Compose([T.NormalizeFeatures(),
                           T.ToUndirected(),
                           T.RandomLinkSplit(
                                num_val=0.2,
                                num_test=0.2,
                                is_undirected=True,
                                add_negative_train_samples=False,
                                # disjoint_train_ratio=0.0,
                                # neg_sampling_ratio=5,
                                edge_types=[edge_type],
                                rev_edge_types=[rev_edge_type]
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
    num_nodes_dict = {ntype: train_data[ntype].num_nodes for ntype in train_data.node_types}
    model = HeteroGNN(metadata=train_data.metadata(),
                      in_channels_dict=in_dims,
                      hidden_channels=256,
                      out_channels=128,
                      dropout=0.2,
                      num_nodes_dict=num_nodes_dict
                      )

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


    # Node counts for negative sampling
    node_counts = {ntype: train_data[ntype].num_nodes for ntype in train_data.node_types}

    epochs = 100
    seed_node_type = None
    check_train_test_overlap(train_data, test_data)

    # --- VAL positives (use edge_label_index) ---
    val_mask = val_data[edge_type].edge_label == 1
    val_ei = val_data[edge_type].edge_label_index[:, val_mask]
    p_val, t_val = val_ei[0], val_ei[1]
    val_pos_edges = list(zip(p_val.tolist(), t_val.tolist()))

    val_gt_tracks = defaultdict(list)
    for p, t in val_pos_edges:
        val_gt_tracks[p].append(t)

    # --- TEST positives (use edge_label_index) ---
    test_mask = test_data[edge_type].edge_label == 1
    test_ei = test_data[edge_type].edge_label_index[:, test_mask]
    p_test, t_test = test_ei[0], test_ei[1]
    test_pos_edges = list(zip(p_test.tolist(), t_test.tolist()))

    test_gt_tracks = defaultdict(list)
    for p, t in test_pos_edges:
        test_gt_tracks[p].append(t)

    # tracks already seen in TRAIN graph
    p_tr, t_tr = train_data[edge_type].edge_index
    seen_tracks = defaultdict(set)
    for p, t in zip(p_tr.tolist(), t_tr.tolist()):
        seen_tracks[p].add(t)
    # -------------------------
    # FULL-GRAPH TRAINING (fixed)
    # -------------------------
    epoch_losses = []

    # Move whole objects to device (simplest way)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Positives (TRAIN graph edges)
    pos_edge_index = train_data[edge_type].edge_index  # [2, E_pos]
    p_pos, t_pos = pos_edge_index[0], pos_edge_index[1]

    num_tracks = train_data["Track"].num_nodes

    # For "is this a real edge?" filtering:
    # NOTE: p_pos/t_pos are on GPU, so .tolist() makes a CPU copy (fine).
    train_pos_set = set(zip(p_pos.tolist(), t_pos.tolist()))
    train_val_pos_set = set(val_pos_edges) | train_pos_set

    K = 100
    best_state = None
    best_val_score = -1e9
    best_epoch = -1
    for epoch in range(1, epochs + 1):


        # -------- train step (full-graph) --------
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x_dict, train_data.edge_index_dict)
        playlist_emb = F.normalize(out["Playlist"], dim=1)
        track_emb = F.normalize(out["Track"], dim=1)

        t_neg = sample_bpr_negatives_train_only(p_pos, num_tracks, train_pos_set, num_neg=50) #[E, num_neg]

        # scores: [E] and [E,K]
        pos_scores = (playlist_emb[p_pos] * track_emb[t_pos]).sum(dim=1)  # [E]
        neg_scores = (playlist_emb[p_pos].unsqueeze(1) * track_emb[t_neg]).sum(2)  # [E,num_neg]

        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [E, num_neg+1]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive is index 0

        loss = F.cross_entropy(logits, labels)

        train_margin = (pos_scores.unsqueeze(1) - neg_scores).mean().item()
        train_auc_approx = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())


        # -------- eval metrics --------
        model.eval()
        # -------- eval metrics (VALIDATION) --------
        model.eval()
        with torch.no_grad():
            out = model(train_data.x_dict, train_data.edge_index_dict)
            playlist_emb = F.normalize(out["Playlist"], dim=1)
            track_emb = F.normalize(out["Track"], dim=1)

            std_t = out["Track"].std(dim=0).mean().item()
            std_p = out["Playlist"].std(dim=0).mean().item()

            # ---- gAUC on VAL (pos vs random negatives) ----
            correct = 0
            total = 0
            for p, t_posi in val_pos_edges:
                pos = torch.dot(playlist_emb[p], track_emb[t_posi]).item()
                neg_drawn = 0
                while neg_drawn < 200:
                    t_n = random.randrange(num_tracks)
                    if (p, t_n) in train_val_pos_set:
                        continue
                    neg = torch.dot(playlist_emb[p], track_emb[t_n]).item()
                    correct += (pos > neg)
                    total += 1
                    neg_drawn += 1
            gauc = correct / max(1, total)

            # ---- MRR / NDCG@K / Hits@K / Recall@K on VAL ----
            mrr = 0.0
            ndcg = 0.0
            hits_k = 0
            recall_k = 0.0
            n = 0

            for p, gt in val_gt_tracks.items():
                sims = (playlist_emb[p:p + 1] @ track_emb.t()).squeeze(0)

                # mask train-seen tracks
                ignore = seen_tracks.get(p, set())
                if ignore:
                    sims = sims.clone()
                    sims[list(ignore)] = -float("inf")

                # top-K (for Hits/Recall)
                topk_idx = torch.topk(sims, K).indices.tolist()
                topk_set = set(topk_idx)
                gt_set = set(gt)

                hit_count = len(topk_set & gt_set)
                if hit_count > 0:
                    hits_k += 1
                recall_k += hit_count / len(gt_set)

                # ranks for MRR/NDCG
                ranks = []
                for t in gt:
                    if torch.isneginf(sims[t]):
                        continue
                    rank = int((sims > sims[t]).sum().item()) + 1
                    ranks.append(rank)

                if not ranks:
                    continue

                best = min(ranks)
                mrr += 1.0 / best

                dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= K)
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), K)))
                ndcg += (dcg / idcg) if idcg > 0 else 0.0

                n += 1

            mrr /= max(1, n)
            ndcg /= max(1, n)
            hits_k /= max(1, n)
            recall_k /= max(1, n)

        # pick a selection metric (common: Recall@K or NDCG@K)
        val_score = recall_k  # or: ndcg

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss: {loss.item():.4f} | "
            f"train_margin: {train_margin:.4f} | "
            f"train_AUC~: {train_auc_approx:.4f} | "
            f"VAL gAUC: {gauc:.4f} | "
            f"VAL MRR: {mrr:.4f} | "
            f"VAL NDCG@{K}: {ndcg:.4f} | "
            f"VAL Hits@{K}: {hits_k:.4f} | "
            f"VAL Recall@{K}: {recall_k:.4f} | "
            # f"StdT: {std_t:.4f} | StdP: {std_p:.4f} | "
            f"best_epoch: {best_epoch:03d} (best_val={best_val_score:.4f})"
        )

    # Plot loss
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.savefig("Loss.png")

    # -------- FINAL TEST (once) --------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    all_pos_set = set(train_pos_set)
    all_pos_set.update(val_pos_edges)
    all_pos_set.update(test_pos_edges)

    model.eval()
    with torch.no_grad():
        out = model(train_data.x_dict, train_data.edge_index_dict)
        playlist_emb = F.normalize(out["Playlist"], dim=1)
        track_emb = F.normalize(out["Track"], dim=1)

        # gAUC on TEST
        correct = 0
        total = 0
        for p, t_posi in test_pos_edges:
            pos = torch.dot(playlist_emb[p], track_emb[t_posi]).item()
            neg_drawn = 0
            while neg_drawn < 200:
                t_n = random.randrange(num_tracks)
                if (p, t_n) in all_pos_set:
                    continue
                neg = torch.dot(playlist_emb[p], track_emb[t_n]).item()
                correct += (pos > neg)
                total += 1
                neg_drawn += 1
        test_gauc = correct / max(1, total)

        # MRR / NDCG / Hits / Recall on TEST
        test_mrr = 0.0
        test_ndcg = 0.0
        test_hits = 0
        test_recall = 0.0
        n = 0

        for p, gt in test_gt_tracks.items():
            sims = (playlist_emb[p:p + 1] @ track_emb.t()).squeeze(0)

            ignore = seen_tracks.get(p, set())
            if ignore:
                sims = sims.clone()
                sims[list(ignore)] = -float("inf")

            topk_idx = torch.topk(sims, K).indices.tolist()
            topk_set = set(topk_idx)
            gt_set = set(gt)

            hit_count = len(topk_set & gt_set)
            if hit_count > 0:
                test_hits += 1
            test_recall += hit_count / len(gt_set)

            ranks = []
            for t in gt:
                if torch.isneginf(sims[t]):
                    continue
                rank = int((sims > sims[t]).sum().item()) + 1
                ranks.append(rank)

            if not ranks:
                continue

            best = min(ranks)
            test_mrr += 1.0 / best

            dcg = sum(1.0 / math.log2(r + 1) for r in ranks if r <= K)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt), K)))
            test_ndcg += (dcg / idcg) if idcg > 0 else 0.0

            n += 1

        n = max(1, n)
        test_mrr /= n
        test_ndcg /= n
        test_hits /= n
        test_recall /= n

    print("\n===== FINAL TEST (best val checkpoint) =====")
    print(f"Best val epoch: {best_epoch} | best_val={best_val_score:.4f}")
    print(f"TEST gAUC: {test_gauc:.4f} | TEST MRR: {test_mrr:.4f} | TEST NDCG@{K}: {test_ndcg:.4f} | "
          f"TEST Hits@{K}: {test_hits:.4f} | TEST Recall@{K}: {test_recall:.4f}")

