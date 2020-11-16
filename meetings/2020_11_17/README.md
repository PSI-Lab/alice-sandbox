
[model_utils/](/model_utils) are copied from last week's work.
Refer to last week's notes on how to use it as a package.

## Stage 1 model training progress

Stage 1 model finished training. Full training progress plot:

![plot/training_progress.png](plot/training_progress.png)

From the overall loss, we can see that ep9 has the lowest validation loss.
Uploaded model and updated `utils_model.py`:


    # trained on random sequence, after fixing y_loop target value bug, ep10,
    # produced by: https://github.com/PSI-Lab/alice-sandbox/tree/35b592ffe99d31325ff23a14269cd59fec9d4b53/meetings/2020_11_10#debug-stage-1-training
    'v0.2': 'ZnUH0A',


Produced by running:

```
python model_utils/plot_training.py --in_log result/rf_data_all_targets_3/run.log  --out_plot result/rf_data_all_targets_3/training_progress.html
```



## Stage 1 model performance




## TODOs

- waiting for stage 1 model to finish training

- stage 1 prevent overfitting (note that theoretical upper bound is not 100% due to the way we constructed the predictive problem)

- upload best model to DC?

- evaluate rfam stage 2 predictions, majority are not identical, but are they close enough?

- investigate pseudoknot predictions, synthetic dataset (45886-32008)

- try running RNAfold and allow C-U and U-U (and other) base pairs, can we recover the lower FE structure that our model predicts?

- rfam151 dataset debug, is the ground truth bounding box correct? (make sure there’s no off-by-1 error)

- stage 1 model: iloop size = 0 on my side is bulge, make sure we have those cases!

- RNAfold performance on rfam151

