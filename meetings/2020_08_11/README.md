
## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

- read capsule network paper

- try simpler conv net (remove res blocks)

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
