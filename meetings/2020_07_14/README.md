


## Literature review

### ﻿Adventures with RNA Graphs

﻿Tamar Schlick

Proposed ﻿RAG (RNA-As-Graphs) framework which partitions a structure into modular units connected in a network.
This ﻿drasticaly reduces the number of possibilities in conformational space,
such that all possible structural motifs can be described explicitly by graph enumeration methods,
since ﻿graph motif space is much smaller than sequence space.

- Existing representation of RNA structure

    - ﻿Tinoco plot

    - circular linked graph (arcs): easy detection of pesudoknots by crossing of arcs

    - ﻿planar graph, with adjacency matrix

    - ﻿mountain plot

    - ﻿ordered labeled tree graph:
    todo plot + notes

    ![plots/labeled_tree_graph.png](plots/labeled_tree_graph.png)

    - ﻿RAG (RNA-As-Graphs): proposed by this author in 2003.
    Tree graph and dual graph.

- Two RAG graphs for RNA structure

    ![plots/rag_graphs.png](plots/rag_graphs.png)

    ![plots/rag_graphs_2.png](plots/rag_graphs_2.png)

    - Tree graph:
    ﻿bulges, junctions, and loops represented as vertices, stems are edges.

    - Dual graph:
    ﻿bulges, junctions, and loops are edges, and stems are vertices.
    Can represent pesudoknot.

- For a structure represented as RAG, we can compute its nxn Laplacian matrix L.
﻿The Laplacian has non-negative eigenvalues ﻿and the second eigenvalue λ2 is a measure of overall compactness.
λ2 was used to order our RNA graphs in the RAG catalog.
The second eigenvector μ2 {μ21, …, μ2n} was used for partitioning graphs.

- Partitioning

    ![plots/graph_partition.png](plots/graph_partition.png)

    - tree graph: graph cut.
    ﻿divides the graph between the two indices k,m whose respective components of μ2 generate the largest numerical difference,
﻿   This partitioning method leaves junctions intact: todo plot

    - dual graph:
    ﻿divides a dual graph into pseudoknot and pseudoknot-free regions.
    todo plot.

- RAG database ﻿enumerated all RNA tree graphs in the RAG framework up to 13 vertices using graph enumeration techniques.

- Also proposed framework for 3D structure. TODO read later.

todo DB notes

they have web tool to convert .ct to graph
http://www.biomath.nyu.edu/?q=rag/rna_matrix

can maybe download the software and run locally:
http://www.biomath.nyu.edu/?q=software/RAGTOP


todo other work on RNA structure partition?


## carry-overs
