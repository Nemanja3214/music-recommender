import pymongo
from collections.abc import MutableMapping
# namedtuple is still in collections, so you might need separate imports
from collections import namedtuple
import certifi
class MongoCache(object):
    def __init__(self):
        with open("api_keys", "r") as f:
            lines = [line.strip() for line in f.readlines()]
        connection_string = lines[3]
        self.client = pymongo.MongoClient(
            connection_string,
            tls=True,
            tlsCAFile=certifi.where(),  # <-- key line
        )
        self.TRACKS_NAME = "tracks"
        self.ARTISTS_NAME = "artists"
        self.db = self.client["music-recommender"]
        self.track_collection = self.db[self.TRACKS_NAME]
        self.artist_genres_collection = self.db[self.ARTISTS_NAME]


    def close(self):
        self.client.close()

    def add(self, id, collection_name, document):
        document["_id"] = id
        collection = self.db[collection_name]
        result = collection.insert_one(document)

    def exists(self, id, collection_name):
        collection = self.db[collection_name]
        doc = collection.find_one({"_id": id})
        return doc is not None

    def get(self, id, collection_name):
        collection = self.db[collection_name]
        doc = collection.find_one(id)
        return doc

    def get_all_ids(self,  collection_name):
        return self.db[collection_name].find({}, {"_id": 1})

