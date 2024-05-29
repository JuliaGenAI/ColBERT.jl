wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz -P downloads/
tar -xvzf downloads/lotte.tar.gz -C downloads/
head downloads/lotte/lifestyle/dev/collection.tsv > downloads/lotte/lifestyle/dev/short_collection.tsv
