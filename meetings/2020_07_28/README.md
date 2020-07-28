

local structure code update:

    - include closing bases in local structure (similar definition as in energy based models),
     with this all connected local structures are sharing the 'corner'

    - fixed plotly shapes so that bounding boxes are surrounding the structures (-0.5 pixel)

    - added pseudo knot detection and bounding box (always on top of off-diagonal, with edge touching the two stems)




For an example on pseudo knot structures, use dataset '../../rna_ss/data_processing/e2efold/data/rnastralign.pkl.gz'
and run script on row index 18 (seq_id: ./RNAStrAlign/16S_rRNA_database/Bacilli/AY074879.ct).
For an example on long range pesudo knot, see index 286 (seq_id: ./RNAStrAlign/group_I_intron_database/IB4/Dpa.cL2500_p1.ct).

See notes in doc: https://docs.google.com/document/d/1IiU89BdAAIB_HW3aLFR6xJA53BB_z_Ym-dDnsLChth0/edit?usp=sharing

todo plot


utils


todo explain local structure array characteristic, give examples + plot
google doc


todo train NN

todo pesudoknot

process one dataset


yolov3: construct toy dataset using RNA structures


yolov3
https://github.com/eriklindernoren/PyTorch-YOLOv3
train reusing their training pipeline before writing my own code?

bounding box resolution?
pixel?
per-pixel classification (as opposed to per grid)?
how to predict bounding box size?
limit local structure size?

per pixel classes in addition to bounding box type:
corner of box, edge of box, inside box?

or: just predict the corners?
only need two corners (uniquely defines a rectangle)

advantage: run on super long sequence, linear time?
conv net for local structure prediction,
discrete opt for assembly,
ensures validity

