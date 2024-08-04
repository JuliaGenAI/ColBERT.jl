using ColBERT
using CUDA

# create the config
dataroot = "downloads/lotte"
dataset = "lifestyle"
datasplit = "dev"
path = joinpath(dataroot, dataset, datasplit, "short_collection.tsv")

nbits = 2   # encode each dimension with 2 bits

index_root = "experiments/notebook/indexes"
index_name = "short_$(dataset).$(datasplit).$(nbits)bits"
index_path = joinpath(index_root, index_name)

# build the searcher
searcher = Searcher(index_path)

# search for a query
query = "what are white spots on raspberries?"
pids, scores = search(searcher, query, 2)
print(searcher.config.resource_settings.collection.data[pids])

query = "are rabbits easy to housebreak?"
pids, scores = search(searcher, query, 9)
print(searcher.config.resource_settings.collection.data[pids])


