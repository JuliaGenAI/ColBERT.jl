using ColBERT
using Test

dataroot = "downloads/lotte"
dataset = "lifestyle"
datasplit = "dev"
path = joinpath(dataroot, dataset, datasplit, "short_collection.tsv")

collection = Collection(path)
length(collection.data)

nbits = 2   # encode each dimension with 2 bits
doc_maxlen = 300   # truncate passages at 300 tokens

checkpoint = "colbert-ir/colbertv2.0"                       # the HF checkpoint
index_root = "experiments/notebook/indexes"
index_name = "short_$(dataset).$(datasplit).$(nbits)bits"
index_path = joinpath(index_root, index_name)

config = ColBERTConfig(
    RunSettings(
        experiment="notebook",
    ),
    TokenizerSettings(),
    ResourceSettings(
        checkpoint=checkpoint,
        collection=collection,
        index_name=index_name,
    ),
    DocSettings(
        doc_maxlen=doc_maxlen,
    ),
    QuerySettings(),
    IndexingSettings(
        index_path=index_path,
        nbits=nbits,
        kmeans_niters=20,
    ),
    SearchSettings(),
)

# trying to load the BaseColBERT
base_colbert = BaseColBERT(checkpoint, config)
checkPoint = Checkpoint(base_colbert, DocTokenizer(base_colbert.tokenizer, config), config)

# getting embeddings and doclens for all passages
bsize = 2
D, doclens = ColBERT.docFromText(checkPoint, collection.data, bsize)

bsize = 3           # should give the same results
new_D, new_doclens = ColBERT.docFromText(checkPoint, collection.data, bsize)

@test isequal(D, new_D)
@test isequal(doclens, new_doclens)
