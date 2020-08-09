## Sementic segmentation

Reading https://www.cs.toronto.edu/~tingwuwang/semantic_segmentation.pdf

- Semantic segmentation before deep learning
    1. relying on conditional random field.
    2. operating on pixels or superpixels
    3. incorporate local evidence in unary potentials
    4. interactions between label assignments

conv net + CRF to refine boundaries:
Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K. and Yuille, A.L., 2014. Semantic image
segmentation with deep convolutional nets and fully connected crfs. arXiv preprint arXiv:1412.7062.

CRF as RNN, end-to-end training:
Zheng, S., Jayasumana, S., Romera-Paredes, B., Vineet, V., Su, Z., Du, D., Huang, C. and Torr, P.H., 2015.
Conditional random fields as recurrent neural networks. In Proceedings of the IEEE International
Conference on Computer Vision.

How do people model semantics e.g. non-local constraints?
This is like discrete opt in our case (stage 2).


Reading https://www.jeremyjordan.me/semantic-segmentation/

For upsampling:
transpose convolution: take a single value from the low-resolution feature map
and multiply all of the weights in our filter by this value,
projecting those weighted values into the output feature map.
For filter sizes which produce an overlap in the output feature map,
the overlapping values are simply added together.
This tends to produce a checkerboard artifact in the output and is undesirable,
so it's best to ensure that your filter size does not produce an overlap.

Semantic segmentation faces an inherent tension between semantics and location:
global information resolves what while local information resolves where...
Combining fine layers and coarse layers lets the model make local predictions
that respect global structure. ― Long et al.
Author proposed to add skip connection from encoder, which improves result.

U-Net architecture:
"consists of a contracting path to capture context and a symmetric expanding path that enables precise localization."
data augmentations ("random elastic deformations of the training samples") as a key concept for learning.

Improvements on U-net:
residual block, dense net


loss:
commonly used: pixel-wise cross entropy loss,
re-weighting each class, re-weighting by distance to segmentation center (boundaries are more important),
soft Dice loss (1 - dice coefficient) on predicting and target mask
(simple sum or squared sum for calculating number of elements within set) (code example in the post),

Reading http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf

Faster R-CNN, region proposal network:
A Region Proposal Network (RPN) takes an image
(of any size) as input and outputs a set of rectangular
object proposals, each with an objectness score.

mask r-cnn, instance segmentation:


## Trying out ideas

### Generic code improvement

- Dataset uploaded to DC: `MVRSGa`

- handles corner case where all elements are being masked.
n_valid_output (denominator) will be 0, which makes the loss NaN.
Since for those cases numerator is 0, wlog, we set the denominator to 1.

- Script to visualize training and validation performance: [visualize_prediction.py](visualize_prediction.py)

To use, e.g.
```
python visualize_prediction.py --in_file result/test_simple_conv_1/pred_ep_9.pkl.gz --out_file result/test_simple_conv_1/plot/pred_plot.html
```

- report loss for different outputs: added simple logging


Updated baseline training script: [train_nn.py](train_nn.py)

### simpler conv net (remove res blocks)

--num_filters 32 32 64 64 128 128 --filter_width 9 9 9 9 9 9

trained on GPU:
```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net.py --data MVRSGa --result result/test_simple_conv_1 --num_filters 32 32 64 64 128 128 --filter_width 9 9 9 9 9 9 --epoch 100 --batch_size 40 --max_length 200 --cpu 16
```

log & plot (git hash in log): https://drive.google.com/drive/folders/1qUlr-Jrcl7OcSg2QCBbct3SZ-K5Fp97W

overfit?
```
(training last minibatch 0.0324, 0.0174, 0.0153, 0.0067, 0.0048)
Epoch 99/100, training loss (running) 0.012379504205740017
(validation last minibatch 0.6953, 0.5342, 0.0854, 0.1866, 0.0240)
Epoch 99/100, validation loss 0.32930181997603386
```

### simpler conv net with dropout

--num_filters 32 32 64 64 128 128 --filter_width 9 9 9 9 9 9 --dropout 0.5

trained on GPU:
```
CUDA_VISIBLE_DEVICES=0 python train_simple_conv_net.py --data MVRSGa --result result/test_simple_conv_2 --num_filters 32 32 64 64 128 128 --filter_width 9 9 9 9 9 9 --dropout 0.5 --epoch 100 --batch_size 40 --max_length 200 --cpu 16
```

running


### Multi-class softmax

Make new dataset:

```
# original sigmoid (non mutually exclusive):
# not_ss, stem, i_loop, h_loop, corner
# mutually exclusive classes correspond to the following assignments:
# 1, 0, 0, 0, 0    # non-local structure region
# 0, 1, 0, 0, 0    # within stem
# 0, 1, 1, 0, 1    # stem - internal_loop boundary
# 0, 1, 0, 1, 1    # stem - hairpin_loop boundary
# 0, 1, 0, 0, 1    # stem - other non-local structure boundary
# 0, 1, 1, 1, 1    # special case where stem length = 1, and is also the boundary between i_loop and h_loop
# 0, 0, 1, 0, 0    # inside internal_loop
# 0, 0, 0, 1, 0    # inside hairpin loop
# 8 classes in total
```


run:
```
python make_data_multiclass.py
```

Upload to DC:
`xCzNlr`


train (debug):
```
python train_simple_conv_net_softmax.py --data xCzNlr --result result/test_simple_conv_softmax_1 --num_filters 16 --filter_width 9 --epoch 10 --batch_size 20 --max_length 80 --cpu 4
```

train (GPU):
todo



## Ideas & TODOs

Make sure to log the conclusion for each idea, for future reference.
(make one section for each idea, move above)
(also for each idea include git hash so we can check the associated training code)

- remove pooling? - we don't have pooling

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
