from rdflib.namespace import RDF
import torch
import numpy as np
import dgl
from sentence_transformers import SentenceTransformer
from music_graph import SpotifyMusicGraphSchema

# ---------------- SentenceTransformer Setup ----------------
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
        # "Genre": "Genre",
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
        data_dict = {}
        max_feature_list_len = 512
        # using canonical because for example byArtist is ambiguous it can be used for albums and tracks
        for (st, pred, ot), edges in self.relations.items():
            # Original edges: [('A', 'B'), ('B', 'C'), ('C', 'A')]
            # after zip:
            # Sources: ('A', 'B', 'C')
            # Destinations: ('B', 'C', 'A')
            src_ids, dst_ids = zip(*edges)
            data_dict[(st, pred, ot)] = (torch.tensor(src_ids), torch.tensor(dst_ids))

        hg = dgl.heterograph(data_dict)

        # for each type
        for ntype in hg.ntypes:
            # get number of nodes for that type
            num_nodes = hg.num_nodes(ntype)
            numeric_dim = 0

            for uri, idx in self.nodes_id_map[ntype].items():
                # get previously extracted numerical features
                feats = self.node_numeric_features.get(uri, {})
                numeric_dim = max(numeric_dim, len(feats))

            # combine numerical and string(this one is fixed and is NAME_DIM) size get dimension,
            # total_dim = numeric_dim + NAME_DIM
            total_dim =max_feature_list_len
            mat = torch.zeros((num_nodes, total_dim), dtype=torch.float32)

            # go through each node
            for uri, idx in self.nodes_id_map[ntype].items():
                feats = self.node_numeric_features.get(uri, {})
                numeric_vec = np.zeros(numeric_dim, dtype=np.float32)
                # extract feat values and put them in vector
                for i, (feat_name, feat_val) in enumerate(feats.items()):
                    if i < numeric_dim:
                        numeric_vec[i] = feat_val
                # concatenate name embedding
                name_vec = embed_name(self.node_names.get(uri, ""))
                combined_vec = np.concatenate([numeric_vec, name_vec])
                combined_vec = np.concatenate([combined_vec, np.zeros(max_feature_list_len - len(combined_vec))])
                mat[idx] = torch.tensor(combined_vec)
            hg.nodes[ntype].data["h"] = mat

        return hg

    def run(self):
        self.collect_rdf_types()
        self.collect_features_and_names()
        self.index_nodes_and_edges()
        hg = self.build_heterograph()
        return hg, self.nodes_id_map

# ---------------- Function to print graph info ----------------
def print_graph_info(hg, node_id_map):
    print("==== Heterograph Info ====")
    for ntype in hg.ntypes:
        print(f"Node type: {ntype}, num_nodes: {hg.num_nodes(ntype)}, feat shape: {hg.nodes[ntype].data['h'].shape}")
        print(f"Features {hg.nodes[ntype].data['h']}")
    for etype in hg.canonical_etypes:
        src, dst = hg.edges(etype=etype)
        print(f"Edge type: {etype}, num_edges: {len(src)}")
    print("Node ID mapping example (first 5 per type):")
    for ntype, mapping in node_id_map.items():
        print(ntype, {k: v for k, v in list(mapping.items())[:5]})


# # ---------------- Run ----------------
# builder = SpotifyHeteroGraphBuilder()
# hg, node_id_map = builder.run()
# print_graph_info(hg, node_id_map)