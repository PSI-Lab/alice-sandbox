

local structure code update:

    - include closing bases in local structure (similar definition as in energy based models),
     with this all connected local structures are sharing the 'corner'

    - fixed plotly shapes so that bounding boxes are surrounding the structures (-0.5 pixel)

    - added pseudo knot detection and bounding box (always on top of off-diagonal, with edge touching the two stems)


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