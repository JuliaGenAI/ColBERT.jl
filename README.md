# ColBERT.jl: Efficient, late-interaction retrieval systems in Julia!

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://codetalker7.github.io/ColBERT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://codetalker7.github.io/ColBERT.jl/dev/)
[![Build Status](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/codetalker7/ColBERT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/codetalker7/ColBERT.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[ColBERT.jl](https://codetalker7/colbert.jl) is a pure Julia package for the ColBERT information retrieval system[^1][^2][^3], allowing developers
to integrate this powerful neural retrieval algorithm into their own downstream tasks. ColBERT (**c**ontextualized **l**ate interaction over **BERT**) has emerged as a state-of-the-art approach for efficient and effective document retrieval, thanks to its ability to leverage contextualized embeddings from pre-trained language models like BERT.

[Inspired from the original Python implementation of ColBERT](https://github.com/stanford-futuredata/ColBERT), with [ColBERT.jl](https://codetalker7/colbert.jl), you can now bring this capability to your Julia applications, whether you're working on natural language processing tasks, information retrieval systems, or other areas where relevant document retrieval is crucial. Our package provides a simple and intuitive interface for using ColBERT in Julia, making it easy to get started with this powerful algorithm.

## Get Started

This package is currently under active development, and has not been registered in Julia's package registry yet. To develop this package, simply clone this repository, and from the root of the package just `dev` it:

    ] dev .

Then, import and use the package as needed. Check out the `examples` folder to see an example of the indexing module. It works with the `lifestyle/dev` split of the [LoTTe dataset](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md). Use the `examples/lotte.sh` script to download the dataset, and extract the first `10` rows of the dataset to work with:

    examples/lotte.sh

The script `example/indexing.jl` contains a minimal working example of how indexing is done. It's as simple as creating a configuration object, and calling the `index` function on it.

    julia --project=examples/ examples/indexing.jl

The script creates an index at a path determined from the configuration you've set up. Check out the example to see more details!

## Key Features

As of now, the package supports the following:

  - Offline indexing documents using embeddings generated from the `"colbert-ir/colbertv2.0"` HuggingFace checkpoint.
  - Compression based on the ColBERTv2[^2] compression scheme, i.e using $k$-means centroids and quantized residuals.

## Contributing

Though the package is in it's early stage, PRs and issues are always welcome! Stay tuned for docs and relevant contribution related information to be added to the repo.

## Stay Tuned

We're excited to continue developing and improving [ColBERT.jl](https://github.com/codetalker7/ColBERT.jl), with the following components to be added soon:

  - A searching, scoring and ranking module. Used to get the top ranked documents for a query.
  - A training module, to be used to pre-train a ColBERT model from scratch.
  - Adding support for multiple GPUs. Currently the package is designed to support only on GPU.
  - Implementation of multiprocessing and distributed training.
  - More utilities to the indexer, like updating/removing documents from the index.
  - PLAID[^3] optimizations.
  - More documentation! The package needs a lot more documentation and examples.
  - Integration with downstream packages like [AIHelpMe.jl](https://github.com/svilupp/AIHelpMe.jl) and [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl). This package can be used as a backend for any information retrieval task.
  - Add support for optimization tricks like [vector pooling](https://www.answer.ai/posts/colbert-pooling.html).

[^1]: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) (SIGIR'20)
[^2]: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) (NAACL'22).
[^3]: [PLAID: An Efficient Engine for Late Interaction Retrieval](https://arxiv.org/abs/2205.09707) (CIKM'22).
