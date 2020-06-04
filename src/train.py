
# TODO: global embedding: features word2vec based on triplet loss
# ⚠ Attention: Not all words need to be embedded, such as co-author name
# should return Y


# TODO: graph generate using linkage weight
#       W(Di, Dj) = sum(w_x) if x is in (Di and Dj)
#       w_x: 1. IDF
#            2. coauthors and organization similarity
#       the compute of w_x is in /src/utils.py

# TODO: local embedding: linkage based clustering
# TODO: #clusters estimation
