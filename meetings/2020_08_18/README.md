## Ideas

### General code improvement

- make mini dataset for debug use (faster data loading):

```
dcl upload -s '{"description": "small dataset for debug use."}' tmp/local_struct.bp_rna.multiclass.small_debug.pkl.gz
```

DC ID: `p2c99k`

- make minibatch workflow, pass in training=T/F flag to streamline code

todo

### Pixel-wise encoding of precise location of bounding box

Most of the time, each pixel can be uniquely assigned to one bounding box.
In the case of closing pair of a loop, it's assigned to both the stem and the loop.
In the rare case where the stem is of length 1, and the stem has 2 loops, one on each side,
the pixel is assigned to 2 loops.
Thus, it can be observed that each pixel can be assigned to:

    - 0 or 1 stem box

    - 0, 1 or 2 internal loop box (we'll ignore the case of 2 internal loop for now since it's rare )

    - 0 or 1 hairpin loop

From the above, we conclude that for each pixel we need at most 4 bounding boxes with unique types
(of course each box can be turned on/off independently, like in CV):

    - stem box

    - internal loop box 1

    - internal loop box 2 (rarely used, ignored for now)

    - hairpin loop box

Using the above formulation, we only need to predict the on/off of each box (sigmoid),
without the need to predict its type (also avoid problem of multiple box permutation).

(since we only predict one iloop, in the case there are both
(which only happens for a single pixel at the boundary, and that pixel also needs to be a len=1 stem),
we set that pixel to belong to the box where it is the top right corner)

To encode the location, we use the top right corner as the reference point,
and calculate the distance of the current pixel to the reference,
both horizontally and vertically.
The idea is to predict it using a softmax over finite classes.
Horizontal distance (y/columns) <= 0, e.g. 0, -1, ..., -9, -10, -10_beyond.
Vertical distance (x/rows) >= 0, e.g. 0, 1, ..., 9, 10, 10_beyond.
Basically we assign one class for each integer distance until some distance away (10in the above example).

To encode the size, we use different number of softmax, depending on the box type:

    - stem: one softmax over 1, ..., 9, 10, 10_beyond, since it's square shaped

    - internal loop: two softmax over 1, ..., 9, 10, 10_beyond, one for height one for width

    - hairpin loop: one softmax over 1, ..., 9, 10, 10_beyond, since it's triangle/square shaped
    (caveat: some hairpin loops are very long)


Alternative: encode distance to lower left corner (hairpin is tricky since it's a triangle).

Also: we should still be able to 'decode' boxes > 10 in size, although with reduced accuracy?


Uploaded to DC: `youYTn`

(above from last week)

small dataset for debug: `p2c99k`

train (using small debug dataset):
```
python train_simple_conv_net_pixel_bb.py --data p2c99k --result result/test_simple_conv_pixel_bb_1 --num_filters 16 --filter_width 9 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```

train on stem pixel bb only for debug:

train (GPU):
```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net_pixel_bb.py --data youYTn --result result/test_simple_conv_softmax_1 --num_filters 32 32 64 64 128 128 --filter_width 9 9 9 9 9 9 --epoch 100 --batch_size 40 --max_length 200 --cpu 16
```





training (all targets):

todo



## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

- use sparse array (numpy? scipy?)

- remove pooling? - we don't have pooling

- improve data loader (taking too much memory)

- read capsule network paper


- re-weight 5 losses

- make 1 softmax out of 5 sigmoid (up tp 2^5, but many combinations are invalid).
This address per-pixel compatibility, but not across pixels.

- soften output. instead of 0/1, come up with a value between 0 and 1, e.g.
distance (w/ decaying function) to center of structure.
Note that this might not help with detecting boundaries/corners which are crucial for assembly.

- train on smaller patches. in theory this should be roughly equivalent to
applying local un-masking, but it's a good baseline to establish anyways.
We can generate patches that center at different types of structures.

- per-pixel prediction of bounding box offsets,
represented as a long softmax. For example,
horizontal offset: -10_beyond, -10, -9, ..., -1, 0, 1, ..., 9, 10, 10_beyond;
vertical offset: similar; class: stem/loop/corner?
(need corner to be a separate class, otherwise has to assign that pixel to one of the class,
maybe the closer one?)
Using +/-10 since most stems are within 10bp (todo: ref).
Using the above output, we can hopefully reconstruct the bounding boxes?
Are they compatible?

- corners are hard to predict, but crucial for stage 2 assembly.
If we try to predict only this output will it work?

- pre-train stage 1 on very short seq, e.g, miRNA or RNAfold generated dataset

## Carry-overs

- smaller input region? localize to the structure? (same as masking most of the background)
local un-mask

- toy example?

- per-channel naive guess & performance

- sample negative mask (make sure to always mask lower triangle)

- report metric on different outputs

- other ways to parametrize output?

- run on GPU/Vector cluster

- if we come up with a scheme to assign probability/class to every pixel,
then we can compute the joint probability.
to avoid exponentially many assembly possibilities,
we can apply cut off to construct an initial set of proposals
before running the discrete step.

- debug: terminal internal loop? (does not make sense)

- discover intrinsic structure within dataset (e.g. families?) unsupervised?

- deep learning assemble proposal?

- formulate as RL?
