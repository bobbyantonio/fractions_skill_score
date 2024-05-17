# FSS-useful-criteria
Python code to accompany paper `How to derive skill from the Fractions Skill Score'

# Installation

Requires Scipy and Numpy. Easiest way to install is using conda; navigate to this directory and run:

`pip install .`

Then in python, for example

```
import numpy as np
from fractions_skill_score import fss, fss_random

# Dummy data, with 10 samples, over a domain of 200x200
fcst_array = np.random.randn(10,200,200)
obs_array = np.random.randn(10,200,200)

# Get scores based on 95th percentile threshold
fss_score = fss(X_o=obs_array, X_f=fcst_array, thr=95, threshold_type='percentile', scale=3)
fss_random_score = fss_random(X_o=obs_array, thr=95, threshold_type='percentile', scale=3)
```



# Producing plots

The plots are produced by running the notebook in notebooks/fss_plots.ipynb


