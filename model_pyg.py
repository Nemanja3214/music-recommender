import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling

from torch_geometric.data import HeteroData
from build_graph_network import SpotifyHeteroGraphBuilder

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

    print_pyg_diagnostics(data)

    transform = T.Compose([T.ToUndirected()])
    data = transform(data)
    print_pyg_diagnostics(data)

    split = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=[("Playlist" ,"Has Track", "Track")],
        rev_edge_types=[('Track', 'rev_Has Track', 'Playlist')]
    )
    train_data, val_data, test_data = split(data)
    print(train_data)

    # Build model from final metadata
    in_dims = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN(metadata=data.metadata(),
                      in_channels_dict=in_dims,
                      hidden_channels=256,
                      out_channels=128,
                      dropout=0.2)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    # Move all data to device
    for ntype in data.node_types:
        if data[ntype].x is not None:
            data[ntype].x = data[ntype].x.to(device)
    for etype in data.edge_types:
        if data[etype].edge_index is not None:
            data[etype].edge_index = data[etype].edge_index.to(device)

    # Node counts for negative sampling
    node_counts = {ntype: data[ntype].num_nodes for ntype in data.node_types}

    batch_size = 10
    device = 'cpu'
    epochs = 50
    num_neighbors = [10, 10]
    neg_ratio = 1
    seed_node_type = None
    target_edge = "Playlist" ,"Has Track", "Track"

    train_edge_label_index = train_data[target_edge].edge_label_index
    train_edge_label = train_data[target_edge].edge_label

    val_edge_label_index = val_data[target_edge].edge_label_index
    val_edge_label = val_data[target_edge].edge_label

    test_edge_label_index = test_data[target_edge].edge_label_index
    test_edge_label = test_data[target_edge].edge_label

    trainloader = LinkNeighborLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        edge_label_index=(target_edge, train_edge_label_index),
        edge_label=train_edge_label,
        num_neighbors=num_neighbors,
    )
    valloader = LinkNeighborLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        edge_label_index=(target_edge, val_edge_label_index),
        edge_label=val_edge_label,
        num_neighbors=num_neighbors,
    )
    testloader = LinkNeighborLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        edge_label_index=(target_edge, test_edge_label_index),
        edge_label=test_edge_label,
        num_neighbors=num_neighbors,
    )
    criterion = nn.TripletMarginLoss(margin=1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_epoch = 0.0
        total_pos_epoch = 0
        total_neg_epoch = 0

        optimizer.zero_grad()
        for batch in trainloader:

            print(batch)
            print(type(batch))
            # out is embedding dict
            selected_batch = batch[target_edge]
            positive_edges_index = selected_batch.edge_label_index[:, selected_batch.edge_label == 1]
            negative_edges_index = selected_batch.edge_label_index[:, selected_batch.edge_label == 0]
            print("POSITIVE" + str(positive_edges_index))
            print("NEGATIVE" + str(negative_edges_index))
            exit()

            out = model(batch.x_dict, batch.edge_index_dict)
            # print(embedding)

            batch_loss = 0.0
            # mask = data["Track"].train_mask
            ground_truth = batch["Playlist" ,"Has Track", "Track"].edge_label
            loss = criterion()
            loss.backward()
            optimizer.step()
            print(float(loss))
    #             edge_index = batch[(src_type, rel, dst_type)].edge_index
    #             if edge_index is None or edge_index.size(1) == 0:
    #                 continue
    #             src_idx = edge_index[0]
    #             dst_idx = edge_index[1]
    #
    #             z_src = embeddings[src_type][src_idx]
    #             z_dst = embeddings[dst_type][dst_idx]
    #
    #             # Positive scores
    #             pos_scores = (z_src * z_dst).sum(dim=-1)
    #             pos_labels = torch.ones_like(pos_scores, device=device)
    #
    #             # Negative sampling
    #             num_pos = pos_scores.size(0)
    #             num_neg = num_pos * neg_ratio
    #             neg_dst_idx = torch.randint(0, node_counts[dst_type], (num_neg,), device=device)
    #             neg_src_idx = src_idx.repeat(neg_ratio)
    #             z_neg_src = embeddings[src_type][neg_src_idx]
    #             z_neg_dst = embeddings[dst_type][neg_dst_idx]
    #             neg_scores = (z_neg_src * z_neg_dst).sum(dim=-1)
    #             neg_labels = torch.zeros_like(neg_scores, device=device)
    #
    #             scores = torch.cat([pos_scores, neg_scores], dim=0)
    #             labels = torch.cat([pos_labels, neg_labels], dim=0)
    #
    #             loss = bce(scores, labels)
    #             total_loss += loss
    #             total_pos_epoch += num_pos
    #             total_neg_epoch += num_neg
    #
    #         if total_loss > 0:
    #             total_loss.backward()
    #             optimizer.step()
    #             total_loss_epoch += total_loss.item()
    #
    #     if epoch % 5 == 0 or epoch == 1:
    #         print(f"Epoch {epoch:03d} | Loss: {total_loss_epoch:.4f} | "
    #               f"pos_edges: {total_pos_epoch} | neg_edges: {total_neg_epoch}")
    #
    # # Compute final embeddings for all nodes
    # model.eval()
    # with torch.no_grad():
    #     final_embeddings = model(
    #         {ntype: data[ntype].x for ntype in data.node_types},
    #         {etype: data[etype].edge_index for etype in data.edge_types}
    #     )
    #     for ntype, emb in final_embeddings.items():
    #         data[ntype].emb = emb.cpu()
    #
    # # Train
    # # data_trained = train_link_reconstruction(data, model,
    # #                                         device=device,
    # #                                         epochs=50,
    # #                                         neg_ratio=1)
    #
    # print("Learned embedding shape for first node type:",
    #       data[data.node_types[0]].emb.shape)
