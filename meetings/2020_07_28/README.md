
## Local structure parsing

local structure code update:

    - include closing bases in local structure (similar definition as in energy based models),
     with this all connected local structures are sharing the 'corner'

    - fixed plotly shapes so that bounding boxes are surrounding the structures (-0.5 pixel)

    - added pseudo knot detection and bounding box (always on top of off-diagonal, with edge touching the two stems)

Wrapped up parser implementation. See [tmp_utils.py](tmp_utils.py) (to be moved to project top level).


For an example on pseudo knot structures, use dataset '../../rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz'
and run script on row index 18 (seq_id: ./RNAStrAlign/16S_rRNA_database/Bacilli/AY074879.ct).
For an example on long range pesudo knot, see index 286 (seq_id: ./RNAStrAlign/group_I_intron_database/IB4/Dpa.cL2500_p1.ct).

See more plots and notes in doc: https://docs.google.com/document/d/1IiU89BdAAIB_HW3aLFR6xJA53BB_z_Ym-dDnsLChth0/edit?usp=sharing


We've validated that the following dataset can be processed successfully:

    - RFam151

    - rnastralign

    - archiveII

Other dataset to be processed (some need 0-based/1-based conversion).



## Other

- construct toy dataset for testing training code

- use existing implementation of yolo v3? https://github.com/eriklindernoren/PyTorch-YOLOv3
Or maybe our setup deviates too much from computer vision problems?

- Bounding box resolution? In CV, we're aiming for maximizing overlap between predicted and ground truth.
But in our problem, we want pixel-level accuracy.
Boundary points are especially important, since they defines which local structures are valid combinations
in the assembly step.
Shall we try per-pixel classification (as opposed to per grid)?

- how to predict bounding box size precisely?

- Do we need to limit local structure size?

- Can we just predict the two corners of each type of bounding box?
We can do it for each pixel.
In addition to assign a class for each pixel, we also predict whether its the top right
or bottom left corner of the bounding box.

- How to deal with pixels inside the box?

- How to deal with pixels in the 'background'.
From a "local structure" perspective, there could be a lot of 'false negatives'
(i.e. plausible local structures that looks like structure on its own,
but didn't get assembled into the final structure due to other more global reasons).
Need to be careful when training! Shall we randomly mask out a portion of the background for backprop in each epoch?


- overall advantage: run on super long sequence in linear time (potentially, depend on how we implement the assembly step)?
conv net for local structure prediction,
discrete opt for assembly,
ensures validity.

