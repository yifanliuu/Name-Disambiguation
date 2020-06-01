
# TODO: global embedding related models

# TODO: local embedding related models

# two layers GCN
# Encoder: g1(Y, A) = A_normed * ReLU(A_normed * Y * W0)* W1
#          A_normed = sqrt(D) * A * 1/sqrt(D)
#          A: adjency matrix, D: degree matrix(diag)
# Decode: g2(Z) = sigmoid(Z'Z)
