# import pixiewalk as pw
#
#
# def build_pixiewalk_graphs(hetero_data):
#     """
#     Returns a dict {(src,rel,dst): PixieGraph}
#     """
#     pw_dict = {}
#
#     for etype in hetero_data.edge_types:
#         src, rel, dst = etype
#         ei = hetero_data[etype].edge_index
#         G = pw.Graph()
#
#         for u, v in ei.t().tolist():
#             G.add_edge(int(u), int(v))
#
#         pw_dict[etype] = G
#
#     return pw_dict
#
# def pixiewalk_sample_neighbors(G: pw.Graph, nodes, walk_length=6, num_walks=3):
#     out = set()
#     for n in nodes:
#         walks = G.random_walk(n, walk_length, num_walks)
#         for w in walks:
#             out.update(w)
#     return list(out)
