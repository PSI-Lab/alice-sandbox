

## Update plot from last week

Fixed bug in calculating sensitivity for plotting:
target bb represented in old data format (top left corner as reference),
but prediction in new data format (top right corner reference),
which resulted in very low sensitivity.

Updated plot:

![plot/s1_performance_param_pair.png](plot/s1_performance_param_pair.png)
Above plot: scatter plot of per-example identical bb sensitivity, for all parameter pairwise comparison.
Each data point is one example.

![plot/s1_performance_histogram.png](plot/s1_performance_histogram.png)
Above plot: histogram of per-example identical bb sensitivity for different parameter setting.


produced by make_plot.ipynb.



old dataset in top left corner format, convert everything to top right?
