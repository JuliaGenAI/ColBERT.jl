# ColBERT.jl: Efficient, late-interaction retrieval systems in Julia!

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://codetalker7.github.io/ColBERT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://codetalker7.github.io/ColBERT.jl/dev/)
[![Build Status](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/codetalker7/ColBERT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGenAI/ColBERT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGenAI/ColBERT.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
<!-- [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) -->

[ColBERT.jl](https://codetalker7/colbert.jl) is a pure Julia package for the ColBERT information retrieval system[^1][^2][^3], allowing developers
to integrate this powerful neural retrieval algorithm into their own downstream tasks. ColBERT (**c**ontextualized **l**ate interaction over **BERT**) has emerged as a state-of-the-art approach for efficient and effective document retrieval, thanks to its ability to leverage contextualized embeddings from pre-trained language models like BERT.

[Inspired from the original Python implementation of ColBERT](https://github.com/stanford-futuredata/ColBERT), with [ColBERT.jl](https://codetalker7/colbert.jl), you can now bring this capability to your Julia applications, whether you're working on natural language processing tasks, information retrieval systems, or other areas where relevant document retrieval is crucial. Our package provides a simple and intuitive interface for using ColBERT in Julia, making it easy to get started with this powerful algorithm.

## Get Started

### Creating the documents

First, install the package. Simply clone this repository, and from the rot of the package just `dev` it:

```julia
julia> ] dev .
```

In this example, we'll index a small collection of `10` documents and run a sample query on it. The documents will just be a simple `Vector{String}`. However, for large datasets, you'll want to save the documents in a file and pass the path of the file to the `ColBERTConfig` (more on this in a bit). Here are the first `10` from the `lifestyle/dev` split of the [LoTTe dataset](https://github.com/stanford-futuredata/colbert/blob/main/lotte.md):

```julia
document_passages = [
   "In my experience rabbits are very easy to housebreak. They like to pee and poop in the same place every time, so in most cases all you have to do is put a little bit of their waste in the litter box and they will happily use the litter box. It is very important that if they go somewhere else, miss the edge or kick waste out of the box that you clean it up well and immediately as otherwise those spots will become existing places to pee and poop. When you clean the box, save a little bit of waste and put it in the cleaned box so it smells right to them. For a more foolproof method, you can get a piece of wood soaked with their urine and put that in the box along with droppings or cage them so that they are only in their litter box for a week. Generally, if I try the first method and find that they are not using only the box on the first day, I go for the litter box only for a week method. The wood block works well if you are moving from a hutch outdoors to a litter box indoors. If you have an indoor cage, you can use the cage itself as the litter box (or attach a litter box to the section of the cage the rabbit has used for waste.) Be sure to use clay or newsprint litter as the other types aren't necessarily good for rabbits. Wood litter is okay if you are sure it isn't fir. The most important thing is to clean anywhere they have an accident. High sided boxes help with avoiding kicking soiled litter out of the box, which is the biggest cause of failure in my experience.",
    "...rabbits can be easily trained to use a litter tray, sometimes with more reliability than your average cat! The natural instinct of a wild rabbit to use one area as its latrine is still apparent in its domestic counterparts. (1) The actual process is very similar to pad training a dog or litter box training a cat. Keep the rabbit confined to a small area while training, move any \"accidents\" to the litter box, and the rabbit will naturally start using that area for its business. The source link has the details. (1) Litter Training Your Rabbit Emma Magnus MSc Association of Pet Behaviour Counsellors apbc.org.uk",
    "It could be a multitude of things. Lack of exercise plays a big role in how your dog acts. If they have a lot of unused energy, they're more likely to act up. Giving him treats or praise will encourage the behavior you're trying to prevent. You want to refrain from treats or praise until he's doing what you want him to do. In the mean time, make it clear to him what you want. You want to be the focus of his attention when you come across a child or another dog. You can do this by keeping him right next to you (never in front of you) by using a short leash. If he tries to pull, tilt the leash upwards. Doing so creates unusual pressure on the bottom of his neck, causing him to look up and see what's going on. If he still won't turn his attention to you, you can forcefully nudge him with the side of your leg until he yields. I've found with my dog sometimes I have to step in front of her and hold her muzzle, forcing her to look at me. It's also important that you remain calm. It's easy to get upset and dogs have the ability of reading our emotions. If you're tense and angry, he'll start tune you out. Source: personal experience with my black lab guided by insight from Cesar Millan",
    "I've had a lot of success with crate training. Dogs won't relieve themselves where they sleep or eat. Keeping them in an enclosed area and bringing them outside when they have to go to relieve themselves teaches them where it's OK to do so. I recommend buying an expandable crate. You want to give them just enough room to stand up and turn around, but not enough that they'll start going in the corner. As they grow you can increase the size of the crate accordingly. The Humane Society has some great info on crate training.",
    "I've seen on the \"Dog Whisperer\" that dogs can sense the anxiety that an owner has. The first thing you need to be aware of is your own anxiety when walking your puppy. As far as your puppy goes, its still a puppy, so fear isn't too unnatural. You will want to acclimate it to walking in urban areas by first training it to walk on a leash well. That means you are walking and your dog happens you be accompanying you. Keep your eyes forward, and heel walk the puppy. If the puppy wanders, give the leash a small yank. As your puppy gets accustomed to this, it should be a lot calmer, and you can try for more noisy environments.",
    "@Paperjam is right: crate training is probably the most effective way to quickly housetrain a puppy. I will also add that routine is incredibly important when house-training a puppy. Take your dog out on a schedule: the same times, every day. Also try to take them out shortly after they eat: puppies might have to go as shortly as 30 minutes after a meal. You want to set your pup up for success. Lastly, let me emphasize that accidents will happen. It's natural, it's frustrating, but it's not the end of the world. What you don't want to do is scold your puppy, or \"rub their nose in it\". Dogs don't really learn that way, and usually just ape your emotions to appease you when they can't figure out what's wrong. It's better to try and prevent such accidents from happening, rather than reacting to one that's already transpired.",
    "You can feed cats raw meat, they're obligate carnivores after all, but human processed meats can introduce other bacterias and contaminates into meat that might not be there otherwise. If you want to do this, which I can understand, then you should introduce the raw meats carefully into their diet and only from a source that you trust, such as local butcher who is following good practices (if they save bones for dogs, it's probably a good sign). While you're doing this, monitor your cat carefully and if there is anything happening that concerns you such as unusual stool, vomiting, etc. then stop immediately and potentially take him/her into a vet for a check up. As an aside, there are some really good food products for cats that don't have as much (or little) filling in them. Spend some time reading the labels and look for products that are all meat or very, very high meat volume. Your cats will appreciate it. :)",
    "Cats are just like dogs in that they need a lot of exercise. More importantly, they also share the same instinct to kill, except to a much greater degree. I would say that your cat needs one or both of these things to happen: More exercise (buy some interactive toys) Something to \"kill\" after exercising (and no, don't buy mice just so it can kill them) What I mean by something to kill is a toy or something for it to chew on after some laser chasing or something. If a cat chases after something but can't actually do anything with it, it can build up frustration and aggression.",
    "Because the dog is so young, I think it is very likely you can slowly accustom her to urban walking. I think a natural response when you see a small dog cowering or wimpering in fear is to console them, or pet them. This is not an effective way to dissuade fear. It is better to ignore your dog and not feed into its fearful emotions. [see below] I think the best bet is to teach your dog that walking in urban areas is fun. Bring treats and toys, walk in a zany unpredictable way, go for short burts of sprints, etc. Your dog will eventually begin to associate urban walks with fun times and an owner in good spirits. This is not universally agreed-upon among trainers. Some believe this to be true, others believe petting a fearful dog does no harm to the training procedure. I personally subscribe to the theory that you should only pet and praise your dog when she is doing what you want her to be doing.",
    "There was a trick Cesar Milan (the Dog Whisperer) used to break a puppy of their fear of living in an urban environment: hold their tail up. When dogs are afraid, their tail tends to go between their legs and their heads bow down. When they're comfortable with their surroundings they'll put their tails upward. Simply (well, not so simply with a dachshund) holding their tail up while walking them might help them to feel more confident. It seems silly but it worked for Cesar. If you have Netflix, I recommend watching episode 2 of The Dog Whisperer: http://movies.netflix.com/WiMovie/The_Very_Best_of_Dog_Whisperer_with_Cesar_Millan/70270440.",
]
```

### The `ColBERTConfig`

To start off, make sure you download a ColBERT checkpoint somewhere in your system; for this example, I'll download the `colbert-ir/colbertv2.0` checkpoint in `$HOME/models`:

    git lfs install
    cd $HOME/models
    git clone https://huggingface.co/colbert-ir/colbertv2.0

The next step is to create a configuration object containing details about all parameters used during indexing/searching using ColBERT. All this information is contained in a type called `ColBERTConfig`. Creating a `ColBERTConfig` is easy; it has the right defaults for most users, and one can change the settings using simple kwargs. In this example, we'll create a config for the collection we just created, and we'll also use [`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl) for GPU support (you can use any GPU backend supported by [Flux.jl](https://github.com/FluxML/Flux.jl))!

```julia
julia>  using ColBERT, CUDA, Random;

julia>  Random.seed!(0)                                                 # global seed for a reproducible index

julia>  config = ColBERTConfig(
            use_gpu = true,
            checkpoint = "/home/codetalker7/models/colbertv2.0",        # local path to the colbert checkpoint
            collection = document_passages,                             # can also provide a local path
            doc_maxlen = 300,                                           # max length beyond which docs are truncated
            index_path = "./short_index/",                              # local directory to save the index in
            chunksize = 2                                               # number of docs to store in a chunk
        );
```

You can read more about a [`ColBERTConfig`](https://github.com/codetalker7/ColBERT.jl/blob/302b68caf0c770b5e23c83b1f204808185ffaac5/src/infra/config.jl#L1) from it's docstring.

### Building the index

Building the index is even easier than creating a config; just build an `Indexer` and call the `index` function:

```julia
julia>  indexer = Indexer(config);

julia>  @time index(indexer)
[ Info: Sampling PIDs for clustering and generating their embeddings.
[ Info: # of sampled PIDs = 7
[ Info: Encoding 7 passages.
[ Info: avg_doclen_est = 178.28572       length(local_sample) = 7
  0.217539 seconds (213.11 k allocations: 7.931 MiB, 0.00% compilation time)
[ Info: Splitting the sampled embeddings to a heldout set.
  0.000832 seconds (8 allocations: 1.229 MiB)
[ Info: Creating 512 clusters.
[ Info: Estimated 1782.8572 embeddings.
[ Info: Saving the index plan to ./short_index/plan.json.
[ Info: Saving the config to the indexing path.
[ Info: Training the clusters.
[ Info: Iteration 1/20, max delta: 0.17846265
[ Info: Iteration 2/20, max delta: 0.13685061
[ Info: Iteration 3/20, max delta: 0.18692705
[ Info: Iteration 4/20, max delta: 0.10748553
[ Info: Iteration 5/20, max delta: 0.10305407
[ Info: Iteration 6/20, max delta: 0.017908525
[ Info: Iteration 7/20, max delta: 0.027008116
[ Info: Iteration 8/20, max delta: 0.023782544
[ Info: Iteration 9/20, max delta: 0.0
[ Info: Terminating as max delta 0.0 < 0.0001
[ Info: Got bucket_cutoffs = Float32[-0.021662371, -0.00015685707, 0.020033525] and bucket_weights = Float32[-0.041035336, -0.009812315, 0.008938393, 0.039779153]
[ Info: avg_residual = 0.029879
  0.018622 seconds (30.88 k allocations: 5.544 MiB)
[ Info: Saving codec to ./short_index/centroids.jld2, ./short_index/avg_residual.jld2, ./short_index/bucket_cutoffs.jld2 and ./short_index/bucket_weights.jld2.
[ Info: Building the index.
[ Info: Encoding 2 passages.
[ Info: Saving chunk 1:          2 passages and 395 embeddings. From passage #1 onward.
[ Info: Saving compressed codes to ./short_index/1.codes.jld2 and residuals to ./short_index/1.residuals.jld2
[ Info: Saving doclens to ./short_index/doclens.1.jld2
[ Info: Saving metadata to ./short_index/1.metadata.json
[ Info: Encoding 2 passages.
[ Info: Saving chunk 2:          2 passages and 354 embeddings. From passage #3 onward.
[ Info: Saving compressed codes to ./short_index/2.codes.jld2 and residuals to ./short_index/2.residuals.jld2
[ Info: Saving doclens to ./short_index/doclens.2.jld2
[ Info: Saving metadata to ./short_index/2.metadata.json
[ Info: Encoding 2 passages.
[ Info: Saving chunk 3:          2 passages and 301 embeddings. From passage #5 onward.
[ Info: Saving compressed codes to ./short_index/3.codes.jld2 and residuals to ./short_index/3.residuals.jld2
[ Info: Saving doclens to ./short_index/doclens.3.jld2
[ Info: Saving metadata to ./short_index/3.metadata.json
[ Info: Encoding 2 passages.
[ Info: Saving chunk 4:          2 passages and 294 embeddings. From passage #7 onward.
[ Info: Saving compressed codes to ./short_index/4.codes.jld2 and residuals to ./short_index/4.residuals.jld2
[ Info: Saving doclens to ./short_index/doclens.4.jld2
[ Info: Saving metadata to ./short_index/4.metadata.json
[ Info: Encoding 2 passages.
[ Info: Saving chunk 5:          2 passages and 320 embeddings. From passage #9 onward.
[ Info: Saving compressed codes to ./short_index/5.codes.jld2 and residuals to ./short_index/5.residuals.jld2
[ Info: Saving doclens to ./short_index/doclens.5.jld2
[ Info: Saving metadata to ./short_index/5.metadata.json
  0.371854 seconds (354.43 k allocations: 24.505 MiB, 20.05% gc time, 2.59% compilation time)
[ Info: Updating chunk metadata and indexing plan
[ Info: Building the centroid to embedding IVF.
[ Info: Saving the IVF.
[ Info: Checking if all index files are saved.
[ Info: Found all files!
  0.629364 seconds (602.34 k allocations: 39.565 MiB, 11.84% gc time, 1.53% compilation time)
true
```

### Searching

Once you've built the index for your collection of docs, it's now time to perform a query search. This involves creating a `Searcher` from the path of the index:

```julia
julia>  searcher = Searcher("short_index");
```

Next, simply feed a query to the `search` function, and get the top-`k` best documents for your query:

```julia
julia>  query = "what was Cesar Milan's trick?";

julia>  @time pids, scores = search(searcher, query, 2)
[ Info: Encoding 1 queries.
  0.014886 seconds (26.30 k allocations: 6.063 MiB, 0.00% compilation time)
([10, 8], Float32[5.9721255, 3.7732823])
```

You can now use these `pids` to see which documents match the best against your query:

```julia
julia> print(document_passages[pids[1]])
There was a trick Cesar Milan (the Dog Whisperer) used to break a puppy of their fear of living in an urban environment: hold their tail up. When dogs are afraid, their tail tends to go between their legs and their heads bow down. When they're comfortable with their surroundings they'll put their tails upward. Simply (well, not so simply with a dachshund) holding their tail up while walking them might help them to feel more confident. It seems silly but it worked for Cesar. If you have Netflix, I recommend watching episode 2 of The Dog Whisperer: http://movies.netflix.com/WiMovie/The_Very_Best_of_Dog_Whisperer_with_Cesar_Millan/70270440.
```

## Key Features

As of now, the package supports the following:

  - Offline indexing of documents using embeddings generated from the `"colbert-ir/colbertv2.0"` (or any other checkpoint supported by [`Transformers.jl`](https://github.com/chengchingwen/Transformers.jl)) HuggingFace checkpoint.
  - Compression/decompression based on the ColBERTv2[^2] compression scheme, i.e using $k$-means centroids and quantized residuals.
  - A simple searching/ranking module, which is used to get the top `k`-ranked documents for a query by computing MaxSim[^1] scores.
  - GPU support using any backend supported by [Flux.jl](https://github.com/FluxML/Flux.jl), both for indexing and searcher.

## Contributing

Though the package is in it's early stage, PRs and issues are always welcome! Stay tuned for docs and relevant contribution related information to be added to the repo.

## Stay Tuned

We're excited to continue developing and improving [ColBERT.jl](https://github.com/codetalker7/ColBERT.jl), with the following components to be added soon (in no order of priority):

  - A training module, to be used to pre-train a ColBERT model from scratch.
  - Adding support for multiple GPUs. Currently the package is designed to support only on GPU.
  - Implementation of multiprocessing and distributed training.
  - More utilities to the indexer, like updating/removing documents from the index.
  - PLAID[^3] optimizations.
  - More documentation! The package needs a lot more documentation and examples.
  - Integration with downstream packages like [AIHelpMe.jl](https://github.com/svilupp/AIHelpMe.jl) and [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl). This package can be used as a backend for any information retrieval task.
  - Add support for optimization tricks like [vector pooling](https://www.answer.ai/posts/colbert-pooling.html).

## Cite us!

If you find this package to be useful in your research/applications, please cite the package:

    @misc{ColBERT.jl,
        author  = {Siddhant Chaudhary <urssidd@gmail.com> and contributors},
        title   = {ColBERT.jl},
        url     = {https://github.com/codetalker7/ColBERT.jl},
        version = {v0.1.0},
        year    = {2024},
        month   = {5}
    }

[^1]: [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) (SIGIR'20)
[^2]: [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) (NAACL'22).
[^3]: [PLAID: An Efficient Engine for Late Interaction Retrieval](https://arxiv.org/abs/2205.09707) (CIKM'22).
