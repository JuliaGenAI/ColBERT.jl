using ColBERT

dataroot = "downloads/lotte"
dataset = "lifestyle"
datasplit = "dev"
path = joinpath(dataroot, dataset, datasplit, "collection.tsv")

collection = Collection(path)

