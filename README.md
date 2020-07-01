# Name-Disambiguation

## Semantic features learning

-   step 1: Generate all text file
-   step 2: Train word2vec model
-   step 4: Sample paper triples
-   step 5: Train FCN by triple loss
-   step 6: Output semantic features

## Non-semantic(Relational) features learning

-   step 1: Generate paperID-authors/paperID-organizations/paperID-venues pairs file ordered by name
-   step 2: Generate heterogeneous information network ordered by name
-   step 3: Train HeGAN ordered by name
-   step 4: Output relational features trained in generator and discriminator

## Cluster

-   step 1: Calculate similarity matrices
-   step 2: Run DBSCAN to cluster
-   step 3: Assign lonelymountains to cluster to form a cluster itself
