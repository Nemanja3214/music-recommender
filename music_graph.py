from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, DC
import matplotlib.pyplot as plt
import networkx as nx

class SpotifyMusicGraphSchema:
    def __init__(self):
        self.g = Graph()
        self.SCHEMA = Namespace("http://schema.org/")
        self.SPOTIFY_PREFIX = "http://example.org/spotify/"
        self.SPOTIFY_PLAYLIST_PREFIX = self.SPOTIFY_PREFIX + "playlist/"

        # Bind namespaces
        self.g.bind("schema", self.SCHEMA)
        self.g.bind("dc", DC)
        self.g.bind("rdfs", RDFS)

    def safe_uri(self, value):
        if isinstance(value, URIRef):
            return value
        if isinstance(value, str):
            return URIRef(value)
        raise ValueError(f"Invalid URI type: {value} ({type(value)})")

    def add_artist(self, uri, name, genre_uris):
        artist = self.safe_uri(uri)
        self.g.add((artist, RDF.type, self.SCHEMA.MusicGroup))
        if name:
            self.g.add((artist, RDFS.label, Literal(name)))
        if genre_uris:
            if type(genre_uris) is not list:
                genre_uris = [genre_uris]
            for genre_uri in genre_uris:
                self.g.add((artist, self.SCHEMA.genre, URIRef(genre_uri)))
        return artist

    def add_album(self, uri, title, artists_uri):
        album = self.safe_uri(uri)
        self.g.add((album, RDF.type, self.SCHEMA.MusicAlbum))
        if title:
            self.g.add((album, DC.title, Literal(title)))
        if artists_uri:
            if type(artists_uri) is not list:
                artists_uri = [artists_uri]
            for artist_uri in artists_uri:
                if artist_uri:
                    self.g.add((album, self.SCHEMA.byArtist, URIRef(artist_uri)))
        return album

    def add_track(self, uri, title, album_uri, artists_uri):
        track = self.safe_uri(uri)
        self.g.add((track, RDF.type, self.SCHEMA.MusicRecording))
        if title:
            self.g.add((track, DC.title, Literal(title)))
        if album_uri:
            self.g.add((track, self.SCHEMA.inAlbum, URIRef(album_uri)))
        if artists_uri:
            if type(artists_uri) is not list:
                artists_uri = [artists_uri]
            for artist_uri in artists_uri:
                if artist_uri:
                    self.g.add((track, self.SCHEMA.byArtist, URIRef(artist_uri)))
        return track

    def add_playlist(self, uri, track_uris):
        playlist = self.safe_uri(uri)
        self.g.add((playlist, RDF.type, self.SCHEMA.MusicPlaylist))

        if track_uris:
            for idx, track_uri in enumerate(track_uris, start=1):
                self.g.add((playlist, self.SCHEMA.Track, URIRef(track_uri)))

        return playlist

    def add_genre(self, uri, name=None):
        genre = self.safe_uri(uri)
        self.g.add((genre, RDF.type, self.SCHEMA.Genre))
        if name:
            self.g.add((genre, RDFS.label, Literal(name)))
        return genre

    def serialize(self, fmt="turtle"):
        return self.g.serialize("./graph_schema.ttl", format=fmt)

    def get_label(self, node):
        label = self.g.value(subject=node, predicate=RDFS.label)
        return str(label) if label else str(node)

    def visualize_rdf_graph(self):
        import math

        G = nx.DiGraph()

        for s, p, o in self.g:
            s_label = self.get_label(s)
            o_label = self.get_label(o) if isinstance(o, URIRef) else str(o)

            if len(s_label) > 25: s_label = s_label[:25] + "..."
            if len(o_label) > 25: o_label = o_label[:25] + "..."

            G.add_node(s_label)
            G.add_node(o_label)

            edge_label = str(p.split('#')[-1] if '#' in str(p) else p.split('/')[-1])
            G.add_edge(s_label, o_label, label=edge_label)

        n = max(len(G.nodes), 1)
        k = 1 / math.sqrt(n)
        pos = nx.spring_layout(G, k=k * 3, iterations=200, seed=42)

        plt.figure(figsize=(18, 12))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=900, edgecolors="black")
        nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, connectionstyle="arc3,rad=0.15")
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_color='darkred', font_size=7,
                                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        plt.title("Spotify Music RDF Graph (Schema.org) Visualization", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("plot_schema.png", dpi=300)
        plt.close()


# ===== Example usage =====
if __name__ == "__main__":
    smg = SpotifyMusicGraphSchema()

    rock = smg.add_genre("spotify:genre:rock", "Rock")
    bts = smg.add_artist("spotify:artist:3Nrfpe0tUJi4K4DXYWgMUX", "BTS", rock)
    be_album = smg.add_album("spotify:album:1ATL5GLyefJaxhQzSPVrLX", "BE", bts)
    track1 = smg.add_track("spotify:track:6rqhFgbbKwnb9MLmUQDhG6", "Dynamite", album_uri=be_album, artists_uri=[bts])
    track2 = smg.add_track("spotify:track:0eGsygTp906u18L0Oimnem", "Butter", album_uri=be_album, artists_uri=bts)
    playlist = smg.add_playlist("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
                                track_uris=[track1, track2])
    smg.visualize_rdf_graph()
    smg.serialize()
