import torch
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dgl.nn.pytorch import HeteroGraphConv, GraphConv, SAGEConv


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]



def construct_negative_graph(graph, k, canonical_etype):
    src_type, edge_type, dst_type = canonical_etype
    src, dst = graph.edges(etype=canonical_etype)

    # Negative sampling
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(dst_type), (len(src) * k,))

    neg_g = dgl.heterograph(
        {canonical_etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
    )

    for ntype in graph.ntypes:
        neg_g.nodes[ntype].data['h'] = graph.nodes[ntype].data['h']

    return neg_g



class HeteroGNN(nn.Module):
    def __init__(self, in_feats_dict, hidden_feats, out_feats, canonical_etypes):
        super().__init__()
        # Linear layer per node type to project to common hidden size

        # HeteroGraphConv for message passing
        hidden_dim1 = 128
        hidden_dim2 = 64
        self.conv1 = HeteroGraphConv({
            canonical_etype: SAGEConv(
                (in_feats_dict[src_type], in_feats_dict[dst_type]),
                hidden_dim1,
                'mean'
            )
            for canonical_etype in canonical_etypes
            for src_type, _, dst_type in [canonical_etype]  # unpack
        })
        self.conv2 = HeteroGraphConv({
            canonical_etype: SAGEConv(
                (hidden_dim1, hidden_dim1),
                hidden_dim2,
                'mean'
            )
            for canonical_etype in canonical_etypes
            for src_type, _, dst_type in [canonical_etype]  # unpack
        })
        # self.conv2 = None
        # print(self.conv1)
        # print(self.conv2)

    def forward(self, g, h_dict):
        # print(h_dict)
        # for key, val in h_dict.items():
        #     print(f"{key}: {val.shape}")

        h = self.conv1(g, h_dict)
        # for key, val in h.items():
        #     print(f"{key}: {val.shape}")
        for ntype in h_dict:
            if ntype not in h:
                h[ntype] = h_dict[ntype]

        h = {k: F.relu(v) for k, v in h.items()}
        # for key, val in h.items():
        #     print(f"{key}: {val.shape}")
        h2 = self.conv2(g, h)
        for ntype in h:
            if ntype not in h2:
                h2[ntype] = h[ntype]
        # for key, val in h.items():
        #     print(f"{key}: {val.shape}")

        return h2

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = HeteroGNN(in_features, hidden_features, out_features, rel_names)
        self.pred = DotPredictor()

    def forward(self, g, neg_g, x, canonical_etype):
        h = self.sage(g, x)
        return h

def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    from build_graph_network import SpotifyHeteroGraphBuilder

    # Build heterograph
    builder = SpotifyHeteroGraphBuilder()
    hetero_graph, node_id_map = builder.run()
    # print(f"There is {hg.nodes['Track']} nodes")

    # Node features dict
    node_features = {ntype: hetero_graph.nodes[ntype].data['h'] for ntype in hetero_graph.ntypes}
    # Automatically get input feature sizes for each node type
    in_feats_dict = {ntype: hetero_graph.nodes[ntype].data['h'].shape[1] for ntype in hetero_graph.ntypes}

    print(in_feats_dict)
    # Example output: {'Track': 10, 'Playlist': 6, 'Album': 8, 'Artist': 12}

    # Target edge type (Playlist -> Track) for link prediction
    target_etype = ('Playlist', 'Has Track', 'Track')
    # print(hetero_graph.nodes['Playlist'].data['h'] )
    print(hetero_graph.canonical_etypes)
    # exit()
    k = 5

    model = Model(in_feats_dict, 128, 64, hetero_graph.canonical_etypes)
    with torch.no_grad():  # Initialize lazy modules.
        neg_graph = construct_negative_graph(hetero_graph, k, target_etype)
        out = model(hetero_graph, neg_graph, node_features, target_etype)

    pred = DotPredictor()

    # after model = Model(...)
    g = hetero_graph
    h = node_features  # same dict you pass to forward

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(50):
        neg_graph = construct_negative_graph(hetero_graph, k, target_etype)
        h = model(hetero_graph, neg_graph, node_features, target_etype)
        pos_score = pred(hetero_graph, h)
        neg_score = pred(neg_graph, h)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"Pos score shape: {pos_score.shape}, Neg score shape: {neg_score.shape}")



