

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



### Image segmentation


####﻿Learning Structured Output Representation using Deep Conditional Generative Models

- Two approaches:

    - x->z -> y, here we assume the prior on z depends on x pθ(z|x)

    - x, z -> y, here we assume the prior is independent of x pθ(z)


- multiple sub networks:

    - recognition network: ﻿qφ(z|x, y)

    - (conditional) prior network: ﻿pθ(z|x)

    - generation network: ﻿pθ(y|x, z)

- "recurrent" connection of initial guess﻿ˆy to the prior network

- evaluation:

    - set z to E\[z|x\]

    - evaluate of conditonal likelihood of y given x, by sampling z using the prior network,
    and compute MC estimate of p(y|x)

    - similar to above, but using importance sampling from qφ(z|x, y)


- proposed method to close gap between recognition network and prior network:
﻿setting the recognition network the same as the prior network, ﻿i.e., qφ(z|x, y) = pθ(z|x).
By doing so we get rid of the KL divergence in the objective.
Author call this:﻿Gaussian stochastic neural network (GSNN).
Overall objective is weighted sum of this and the original objective.


![plot/cvae_paper_s1.png](plot/cvae_paper_s1.png)

- Model architeture:

    - recognition network input: original image + ground truth y

    - prior network input: original image + predicted y from baseline CNN

    - final output is summation of generation network and baseline CNN



####﻿A Probabilistic U-Net for Segmentation of Ambiguous Images

![plot/unet_cvae_paper_1.png](plot/unet_cvae_paper_1.png)

- loss is a weighted combination of:

    - KL divergence between probabilities parameterized by prior (condition on x)
    and posterior (condition on x and ground truth y) network

    - likelihood of y|x generated from z sampled from posterior network


### MNIST conditioned on image


#### Add CNN before conditioning

One layer CNN before combining x with y or z.

https://docs.google.com/presentation/d/1XXCik3yw_ww_2h8rPjfEhzn4x4VhH3Oimv5d6HM2BYM/edit#slide=id.gc11059b81d_0_0


#### Add prior network

- In addition to posterior network: x, y -> z's parameter,
add prior network that maps x to z's parameter

- Minimize KL between posterior and the parameterized prior (as opposed to the fixed N(0,1) prior)

- KL divergence between prior and posterior, when both are parameterized:
https://jamboard.google.com/d/1h3RJg4gkF48me63UrBfoUMALqjxkw3bqtcPAVszGc4M/viewer?f=0

- At test time, use prior network to sample z


tie weights?

Can we think of a better toy problem?





## CVAE adopted to bounding box prediction

- use both prior and posterior network

### Generate dataset


Generate multiple y's for the same x by sampling RNAfold suboptimal structure: `RNAsubopt`.
For now we're using the `-p` option: randomly draw structures according to their probability in the Boltzmann ensemble.
There is another (mutually exclusive) option `-e` for generating all structures within a delta free energy threshold.

```
mkdir data/
cd s1_vae_data_gen/
python make_data.py --minlen 10 --maxlen 100 --num_seq 1000 --num_sample 10 --out ../data/synthetic_bb_dist.len10_100.num1000.sample10.pkl.gz
```




