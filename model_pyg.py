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

import torch_sparse
print(torch_sparse.__version__)

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

        self.post_lin = nn.ModuleDict()
        for ntype in self.node_types:
            self.post_lin[ntype] = nn.Linear(out_channels, out_channels)
        print(self)

    def forward(self, x_dict, edge_index_dict):
        # print("DATA ===================================================")
        # print(x_dict)
        x_dict = self.conv1(x_dict, edge_index_dict)
        # print("CONV1 ===================================================")
        # print(x_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        # x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        # print("CONV2 ===================================================")
        # print(x_dict)
        # TODO add residual connection
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}


        out = {}
        for ntype, emb in x_dict.items():
            out[ntype] = self.post_lin[ntype](emb)
        # print("OUT ===================================================")
        # print(out)
        return out


# -------------------------
# Main usage
# -------------------------
if __name__ == "__main__":
    builder = SpotifyHeteroGraphBuilder()
    data, nodes_id_map = builder.run()

    # print_pyg_diagnostics(data)

    transform = T.Compose([T.ToUndirected()])
    data = transform(data)
    print_pyg_diagnostics(data)

    split = T.RandomLinkSplit(
        num_val=0.0,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=70,
        edge_types=[("Playlist" ,"Has Track", "Track")],
        rev_edge_types=[('Track', 'rev_Has Track', 'Playlist')]
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

    batch_size = 30
    device = 'cpu'
    epochs = 100
    num_neighbors = [10, 10]
    seed_node_type = None
    target_edge = "Playlist" ,"Has Track", "Track"

    train_edge_label_index = train_data[target_edge].edge_label_index
    train_edge_label = train_data[target_edge].edge_label
    print(f"LABELS {train_edge_label}")

    # val_edge_label_index = val_data[target_edge].edge_label_index
    # val_edge_label = val_data[target_edge].edge_label

    test_edge_label_index = test_data[target_edge].edge_label_index
    test_edge_label = test_data[target_edge].edge_label

    # def check_edges_overlap():
    #     train_edge_type = ('Playlist', 'Has Track', 'Track')
    #     rev_train_edge_type = ('Track', 'rev_Has Track', 'Playlist')
    #
    #     # Original train edges
    #     train_edges = set(
    #         tuple(x.tolist()) for x in train_data[train_edge_type].edge_label_index.t()
    #     )
    #
    #     # Add reversed edges
    #     rev_train_edges = set(
    #         (dst.item(), src.item())
    #         for src, dst in train_data[rev_train_edge_type].edge_label_index.t()
    #     )
    #     all_train_edges = train_edges.union(rev_train_edges)
    #
    #     test_edge_type = ('Playlist', 'Has Track', 'Track')
    #     rev_test_edge_type = ('Playlist', 'rev_Has Track', 'Track')
    #
    #     test_edges = set(
    #         tuple(x.tolist()) for x in test_data[test_edge_type].edge_label_index.t()
    #     )
    #
    #     # Include reversed test edges
    #     rev_test_edges = set(
    #         (dst.item(), src.item())
    #         for src, dst in test_data[rev_test_edge_type].edge_label_index.t()
    #     )
    #     all_test_edges = test_edges.union(rev_test_edges)
    #
    #     overlap = all_train_edges.intersection(all_test_edges)
    #     print(f"Number of overlapping edges (including reversed): {len(overlap)}")
    #
    #     if len(overlap) > 0:
    #         print("Warning: Some test edges are also in the training set!")
    #     else:
    #         print("No overlap between train and test edges. ✅")
    #
    # check_edges_overlap()

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
            z_p = F.normalize(out['Playlist'][p_idx], dim=1)
            z_t = F.normalize(out['Track'][t_idx], dim=1)

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

        # if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {total_loss_epoch:.4f}"
              f"Skipped {skipped_count}/{batch_count}")
        epoch_losses.append(total_loss_epoch)
    # epoch_losses = [math.log(x) for x in epoch_losses]


    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    plt.savefig("Loss.png")

    print(f"STD track: {out['Track'].std(dim=0).mean()}")
    print(f"STD playlist: {out['Playlist'].std(dim=0).mean()}")

    model.eval()
    with torch.no_grad():
        # Playlist embeddings (all playlists)
        graph_input = {"nodes": {ntype: data[ntype].x.to(device) for ntype in data.node_types},
                "edges": {etype: data[etype].edge_index.to(device) for etype in data.edge_types}}

        track_ids = torch.tensor(list(data["Track"].ids), dtype=torch.long)

        out = model(graph_input["nodes"], graph_input["edges"])
        playlist_emb = out['Playlist']

        # Track embeddings (only those in training set)
        track_emb = out["Track"]

        track_emb = F.normalize(track_emb, dim=1)

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
            p_vec = F.normalize(p_vec, dim=1)

            # similarity with training tracks
            sims = torch.matmul(p_vec, track_emb.t()).squeeze(0)  # [num_train_tracks]

            # top-K recommended track indices (relative to train tracks)
            topk_vals, topk_index = torch.topk(sims, K)
            topk_index = topk_index.tolist()

            # Map topk indices back to global track IDs if needed
            topk_track_ids = track_ids[topk_index].tolist()

            # Hit Rate@K: at least one ground-truth track in top-K
            if any(tid in topk_track_ids for tid in gt_tids):
                hit_count += 1

            # Recall@K: fraction of ground-truth tracks in top-K
            recall_sum += sum(tid in topk_track_ids for tid in gt_tids) / len(gt_tids)

        hit_rate = hit_count / num_playlists
        recall_at_k = recall_sum / num_playlists

        print(f"Hit Rate@{K} (restricted to training tracks): {hit_rate:.4f}")
        print(f"Recall@{K} (restricted to training tracks): {recall_at_k:.4f}")
