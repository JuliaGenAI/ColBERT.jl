# Basic Retrieval Example

This tutorial demonstrates how to use ColBERT.jl for simple document retrieval.

---

## Step 1: Prepare Dataset

The dataset should be in TSV format:

doc_id \t title \t body

Example:

1    Deep Learning    Neural networks are powerful  
2    Machine Learning    Supervised learning is common  

---

## Step 2: Build Index

```julia
using ColBERT

config = ColBERTConfig(
    collection="sample.tsv",
    index_path="index"
)

indexer = Indexer(config)
index(indexer)

## Step 3: Retrieval

After building the index, it can be used for efficient document retrieval.

The querying interface may vary depending on the current implementation.
Users can refer to the latest examples in the repository for performing search.