

## CVAE examples

### MNIST conditioned on class label

Sanity check it works



see predictions: https://docs.google.com/presentation/d/1XXCik3yw_ww_2h8rPjfEhzn4x4VhH3Oimv5d6HM2BYM/edit#slide=id.p


Model:

```
CVAE(
  (fc1): Linear(in_features=794, out_features=200, bias=True)
  (fc21): Linear(in_features=200, out_features=10, bias=True)
  (fc22): Linear(in_features=200, out_features=10, bias=True)
  (fc3): Linear(in_features=20, out_features=200, bias=True)
  (fc4): Linear(in_features=200, out_features=784, bias=True)
)
```

Optimization:

```
Train Epoch: 0 [0/60000 (0%)]	Loss: 10986.682617 = 10985.231445 + 1.451528
Train Epoch: 0 [2000/60000 (3%)]	Loss: 4360.748535 = 4240.848145 + 119.900520
Train Epoch: 0 [4000/60000 (7%)]	Loss: 3583.112793 = 3400.864990 + 182.247910
...
Train Epoch: 3 [54000/60000 (90%)]	Loss: 2169.500488 = 1847.556763 + 321.943817
Train Epoch: 3 [56000/60000 (93%)]	Loss: 2199.115723 = 1896.604614 + 302.511108
Train Epoch: 3 [58000/60000 (97%)]	Loss: 2363.539551 = 2045.863770 + 317.675720
====> Test set loss: 2173.4973
```


### MNIST conditioned on image

Can we think of a better toy problem?


### Image segmentation

Paper: ﻿A Probabilistic U-Net for Segmentation of Ambiguous Images



## CVAE adopted to bounding box prediction



