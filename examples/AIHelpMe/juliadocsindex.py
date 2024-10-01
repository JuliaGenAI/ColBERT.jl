import time

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Collection
from colbert import Indexer, Searcher

# creating the collection
doc_passages = []
doc_sources = []

import h5py

filenames = [
    "genie__v20240818__textembedding3large-1024-Bool__v1.0.hdf5",
    "JuliaData-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "julialang__v20240819__textembedding3large-1024-Bool__v1.0.hdf5",
    "Makie-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "pack.hdf5",
    "Plots-text-embedding-3-large-1-Bool__v1.0.hdf5",
    "sciml__v20240716__textembedding3large-1024-Bool__v1.0.hdf5",
    "tidier__v20240716__textembedding3large-1024-Bool__v1.0.hdf5",
]
for filename in filenames:
    with h5py.File(filename, "r") as file:
        doc_passages += list(file["chunks"])
        doc_sources += list(file["sources"])

# convert to string
for idx in range(len(doc_passages)):
    doc_passages[idx] = doc_passages[idx].decode()
    doc_sources[idx] = doc_sources[idx].decode()

collection = Collection(data=doc_passages)
f"Loaded {len(collection):,} passages"
print(collection[89852])

# build the index
nbits = 2  # encode each dimension with 2 bits
doc_maxlen = 300  # truncate passages at 300 tokens

checkpoint = "/home/chaudhary/models/colbertv2.0"
index_name = f"juliadocsindex_python"

start = time.process_time()
with Run().context(
    RunConfig(nranks=1, experiment="notebook")
):  # nranks specifies the number of GPUs to use.
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits)

    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection, overwrite=True)
time_elapsed = time.process_time() - start
# about 12 minutes; it's much faster

# some sample search queries
with Run().context(RunConfig(experiment="notebook")):
    searcher = Searcher(index=index_name, collection=collection)
query = "How can you construct an uninitialized CategoricalArray with specific levels and dimensions?"
results = searcher.search(query, k=10)
for passage_id, passage_rank, passage_score in zip(*results):
    print(
        f"\t [{passage_rank}] \t\t {passage_score:.1f} \t\t {searcher.collection[passage_id]}"
    )

# evaluation
import json
import pylcs


def distance_longest_common_subsequence(input1, input2):
    if len(input1) == 0 or len(input2) == 0:
        return 1.0
    similarity = pylcs.lcs(input1, input2)
    shortest_length = min(len(input1), len(input2))
    return 1.0 - similarity / shortest_length


def distance_longest_common_subsequence_multiple(input, inputs):
    return [
        distance_longest_common_subsequence(input, input2) for input2 in inputs
    ]


with open("qa_evals.json", "r") as file:
    eval_qa = json.load(file)
k = 5
num_hits = 0
for query in eval_qa:
    pids, _, scores = searcher.search(query["question"], k=k)
    if (
        min(
            distance_longest_common_subsequence_multiple(
                query["context"], [doc_passages[pid] for pid in pids]
            )
        )
        < 0.33
    ):
        num_hits += 1
print("Number of hits: ", num_hits / len(eval_qa))
## Number of hits:  0.8676470588235294 
