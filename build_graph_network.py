from rdflib.namespace import RDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from music_graph import SpotifyMusicGraphSchema
from torch_geometric.data import HeteroData

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
NAME_DIM = MODEL.get_sentence_embedding_dimension()

def embed_name(name: str):
    if not name:
        return np.zeros(NAME_DIM, dtype=np.float32)
    emb = MODEL.encode([name], convert_to_numpy=True, normalize_embeddings=False)
    return emb[0].astype(np.float32)

# ---------------- Spotify Heterograph Builder ----------------
class SpotifyHeteroGraphBuilder:
    TYPE_MAP = {
        "MusicRecording": "Track",
        "MusicPlaylist": "Playlist",
        "MusicGroup": "Artist",
        # "MusicAlbum": "Album",
        "Genre": "Genre",
    }

    def __init__(self):
        self.smg = SpotifyMusicGraphSchema("./graph_schema.ttl")
        self.g = self.smg.g
        self.FEATURES_PREFIX = str(self.smg.FEATURES_PREFIX)
        self.SCHEMA = self.smg.SCHEMA

        self.entity_type = {}
        self.nodes_id_map = {}
        self.relations = {}
        self.node_numeric_features = {}
        self.node_names = {}

    # only looks for rdf type predicates
    def collect_rdf_types(self):
        for s, p, o in self.g:
            if p == RDF.type:
                type_name = str(o).split("/")[-1]

                if type_name in self.TYPE_MAP:
                    self.entity_type[str(s)] = self.TYPE_MAP[type_name]
        # initialize dictionary for each type
        for nt in set(self.TYPE_MAP.values()):
            self.nodes_id_map[nt] = {}

    def collect_features_and_names(self):
        for s, p, o in self.g:
            su = str(s)
            ps = str(p)
            if su not in self.entity_type:
                continue
            # numerical feauters start with FEATURES_PREFIX
            if ps.startswith(self.FEATURES_PREFIX):
                fname = ps.replace(self.FEATURES_PREFIX, "")
                self.node_numeric_features.setdefault(su, {})
                try:
                    # separate dict for numerical
                    self.node_numeric_features[su][fname] = float(o)
                except:
                    pass
            # names are also features
            elif p == self.SCHEMA.Name:
                self.node_names[su] = str(o)

    def index_nodes_and_edges(self):
        for s, p, o in self.g:
            su, ou = str(s), str(o)
            ps = str(p)
            # not interested in type, feature... relationships
            if ps.startswith(self.FEATURES_PREFIX) or p == self.SCHEMA.Name:
                continue
            # not interested in all other relationships than those that contain those between enitity types
            if su not in self.entity_type or ou not in self.entity_type:
                continue
            st = self.entity_type[su]
            ot = self.entity_type[ou]

            # generate nex id based on length of current elements
            # eg we have 2 elements now is 3. we have ids 0,1 now is len(elements) so 2
            if su not in self.nodes_id_map[st]:
                self.nodes_id_map[st][su] = len(self.nodes_id_map[st])
            if ou not in self.nodes_id_map[ot]:
                self.nodes_id_map[ot][ou] = len(self.nodes_id_map[ot])

            # TODO added
            ps = ps.split("/")[-1]
            if ps == "Track":
                ps = "Has Track"
            key = (st, ps, ot)
            # edges in the format (source_type, predicate, dest_type) -> list of (src_id, dst_id)
            if key not in self.relations:
                self.relations[key] = []
            self.relations[key].append((self.nodes_id_map[st][su], self.nodes_id_map[ot][ou]))

    def build_heterograph(self):
        data = HeteroData()
        max_feature_list_len = 512

        # -------------------------
        # 1. Add edges (PyG style)
        # -------------------------
        for (st, pred, ot), edges in self.relations.items():
            if not edges:
                continue
            src_ids, dst_ids = zip(*edges)
            edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            # canonical hetero key in PyG: (src_type, relation, dst_type)
            data[(st, pred, ot)].edge_index = edge_index

        # -------------------------
        # 2. Add node features
        # -------------------------
        for ntype, id_map in self.nodes_id_map.items():
            num_nodes = len(id_map)
            numeric_dim = 0

            # find max numeric feature dimension
            for uri in id_map.keys():
                feats = self.node_numeric_features.get(uri, {})
                numeric_dim = max(numeric_dim, len(feats))

            total_dim = max_feature_list_len
            mat = torch.zeros((num_nodes, total_dim), dtype=torch.float32)

            for uri, idx in id_map.items():
                feats = self.node_numeric_features.get(uri, {})
                numeric_vec = np.zeros(numeric_dim, dtype=np.float32)

                # fill numeric features
                for i, (_, feat_val) in enumerate(feats.items()):
                    if i < numeric_dim:
                        numeric_vec[i] = feat_val

                # embed name
                name_vec = embed_name(self.node_names.get(uri, ""))

                # combine
                combined_vec = np.concatenate([numeric_vec, name_vec])
                if len(combined_vec) < max_feature_list_len:
                    combined_vec = np.concatenate([
                        combined_vec,
                        np.zeros(max_feature_list_len - len(combined_vec), dtype=np.float32)
                    ])
                else:
                    combined_vec = combined_vec[:max_feature_list_len]

                mat[idx] = torch.tensor(combined_vec)

            # assign features to PyG node type
            data[ntype].x = mat
            data[ntype].ids = id_map.values()
            data[ntype].num_nodes = num_nodes

        return data

    def run(self):
        self.collect_rdf_types()
        self.collect_features_and_names()
        self.index_nodes_and_edges()
        hg = self.build_heterograph()
        return hg, self.nodes_id_map



# ---------------- Function to print graph info ----------------
def print_graph_info(hg: HeteroData, node_id_map):
    print("==== Heterograph Info (PyG) ====")

    # ---- Node types ----
    for ntype in hg.node_types:
        x = hg[ntype].get("x", None)
        num_nodes = hg[ntype].num_nodes
        print(f"\nNode type: {ntype}")
        print(f"  num_nodes: {num_nodes}")
        if x is not None:
            print(f"  feature shape: {tuple(x.shape)}")
        else:
            print("  feature shape: None")

    # ---- Edge types ----
    print("\n---- Edge Types ----")
    for etype in hg.edge_types:
        edge_index = hg[etype].edge_index
        num_edges = edge_index.size(1)
        print(f"Edge type: {etype}  num_edges: {num_edges}")

    # ---- Example node ID mappings ----
    print("\n---- Node ID mapping (first 5 entries) ----")
    for ntype, mapping in node_id_map.items():
        preview = list(mapping.items())[:5]
        print(f"{ntype}: {preview}")



# # ---------------- Run ----------------
builder = SpotifyHeteroGraphBuilder()
hg, node_id_map = builder.run()
print_graph_info(hg, node_id_map)