using ColBERT
using CUDA

# build the searcher
index_path = "short_collection_index"
searcher = Searcher(index_path)

# load the collection
collection = readlines(searcher.config.collection)

# search for a query
query = "what are white spots on raspberries?"
pids, scores = search(searcher, query, 2)
print(collection[pids])

query = "are rabbits easy to housebreak?"
pids, scores = search(searcher, query, 1)
print(collection[pids])
