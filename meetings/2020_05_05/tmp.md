ICLR 2020

Tuesday

## Biology and ML

### ﻿RNA SECONDARY STRUCTURE PREDICTION BY LEARNING UNROLLED ALGORITHMS

Interesting work that uses an alternative approach to model output in space with constraint.
Instead of representing the output in a data structure that reflect the constraint,
the author chose to represent it in a larger space,
then use convex optimization to project the output onto the restricted space.
Furthermore, they unrolled the optimization with fixed number of steps as an RNN,
such that parameters can be learned end-to-end.

- Predicting RNA secondary structure from sequence.
Input: RNA sequence of length L, output: binary matrix LxL,
where 1 at (i, j) indicates that i and j are paired.
The author considered 3 constraints that need to be built into the output:

    1. only allowing these bases to pair up: G-C, A-U and G-U

    2. no sharp pairs, i.e. output is 0 if abs(i-j) < 4

    3. one base can only be paired with at most on other base, i.e. output matrix row/column sum <= 1

- Model consists of two parts: deep score network and post-processing network

- Deep score network:

    - input: Lx4 1-hot encoded sequence & absolute and relative position matrix Lx2

    - 1-hot sequence Lx4 converted to Lxd by encoding matrix W

    - position embedding converts Lx2 to Lxd, using feature maps (sin, cos, sigmoid, etc.) followed by MLP

    - sequence and position encoding, concatenated to Lx2d and pass through a series of transformer encoder

    - output of transformer encoder, concatenated with position embedding (Lx3d), form outer product with itself to get 2D feature map: LxLx6d

    - 2D feature map pass though a series of 2D conv, output is matrix: LxLx1


- Post-processing network:


    - formulation of constrained optimization

        - given output of scoring network U, want to find binary matrix A that maximize the total score,
        where total score is a sum of element-wise scores. With scalar offset s,
        for each element where a_ij = 1, it gets a score of (u_ij - s).

        - convex relaxation of binary A to continuous range a_ij within \[0, 1\]

        - define binary matrix M, same shape as U, that masks as 0 invalid entries according to constraint
         1 (entries corresponding to non-(G-C or A-U or G-U) pairs set to 0) and
          2 (entries within distance 4 to the off-diagonal set to 0).

        - all constraints can be incorporated by first transforming A to be
        non-negative and symmetric,
        followed by element-wise multiplication of M to satisfy 1 and 2,
        then adding additional row-sum < 1 as optimization constraint to satisfy 3.

        - L1 penalty added to encourage sparsity

        - constrained optimization formulation in Eq 1

        - convert to a unconstrained optimization in Eq 2, solved by primal-dual method

    - unroll above algorithm to a NN

        - Each update is represented one RNN cell

        - unroll for a fixed number of iterations (RNN steps)

- Training

    - differentiable negative F1 loss, see Eq 8

    - pre-train scoring network with cross entropy

    - jointly train scoring and post-processing network using a loss that incorporate all
    intermediate output during the optimization steps, with discounting factor that emphasize those
    closer to the output.

    - dataset: RNAStralign (﻿30451 sequences after de-dup) & ﻿ArchiveII (3975 sequences)

    - split of training/testing are based on RNA types,
    i.e. same RNA type exists in both training and testing


- concerns:

    - non-efficient 2D convolution: since input to conv is outer product of 1D feature map,
    for a conv sliding window, any entries outside of the middle row/column are redundant.

    - since it's unrolled optimization, output is not guaranteed to satisfy all constraints

    - experiment and evaluation were based on split-on-RNA-types, so we don't know whether
    model generalized to unseen RNA types/families (put aside human mRNA)


### ﻿RECONSTRUCTING CONTINUOUS DISTRIBUTIONS OF 3D PROTEIN STRUCTURE FROM CRYO-EM IMAGES

- task: reconstruct protein 3D structure from ﻿cryo-EM images:
10^(4-7) noisy 2D projection images per each protein,
where each image captures an unknown orientation of the protein
(with potential in-plane translation, i.e. shifted center),
and each pixel in the image represents electron density integrated along the imaging axis.

- this work aims at recover ﻿structural heterogeneity, with the assumption that
protein conformations are continuous in latent space

- image pixel density can be formulated as integration of the structure (volume V) along the unknown
orientation, see Eq 1.
Making use of the "Fourier slice theorem": the Fourier transform of a 2D projection of V is a
2D slice through the origin of V in the Fourier domain,
where the slice is perpendicular to the projection direction.
Under the assumption that frequency-dependent noisy is independent and 0-mean,
probability of image, given structure/volume and pose/orientation, can be formulated in
Fourier domain as in Eq 3.

- proposed model is a VAE,
 where the encode maps 2D image in Fourier space to volume latent variable z,
 and the decode takes z and 3D coordinate
 (from ﻿rotating a DxD lattice on the x-y plane by R, the image orientation latent variable)
  to construct the image pixel by pixel, see Fig 2.
The volume latent variable is estimated by variational inference,
and orientation latent variable is estimated by global search on maximum likelihood solution using Branch-and-Bound.


### ﻿ENERGY-BASED MODELS FOR ATOMIC-RESOLUTION PROTEIN CONFORMATIONS

- terminology: rotamers possible configurations of the side chains within a folded protein

- task: learning energy function of side chain + context atoms

- input: a set of 64 nearest neighbour atoms of the amino acid residual of interest,
Each atom is represented by 256-dim feature vector:
3D Cartesian coordinate embedded in 172-dim,
identity of atom (N, C, O, S), embedded in 28-dim,
ordinal label of the atom in side chain embedded in 28-dim,
amino acid type embedded in 28-dim.
\[It's unclear why they need to do the embedding, other than matching transformer layer size\]

- output: scalar energy

- model: 6-layer transformer followed by 2-layer fully connected. Trained with dropout and layer normalization.
\[it's unclear how the model operate on input as 'set', was the function permutation invariant?\]

- ﻿During training, a random rotation is applied to the coordinates in order to encourage rotational invariance of the model.

- model trained using importance sampling, where the sampler is a collection of rotamers conditioned on the given amino acid and backbone angles.




