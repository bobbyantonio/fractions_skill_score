# fractions-SSIM
Python code for calculating the Fractions-SSIM score

# Installation

Requires Scipy and Numpy. Easiest way to install is using pip; navigate to this directory and run:

`pip install .`

Then in python:

```
from fssim import calculate_fractions_SSIM

fssim_score, min_threshold = calculate_fractions_SSIM(X_o=obs_array, X_f=fcst_array, thr=0.1, scale=3)
```


Input args to `calculate_fractions_SSIM`:

| Argument       | Type | Description |
| ----------- | ----------- |----------- |
| X_o      | np.ndarray      | Array of shape (s, m, n) containing s samples of the observation field       |
| X_f   | np.ndarray         |Array of shape (s, m, n, en) containing an ensemble of c forecast fields, s samples, or (s,m,n) if just one ensemble member (will be converted to the right shape)        |
| scale | int | Size of neighbourhood around each pixel to calculate the fraction.
| thr | float | Threshold to convert rainfall values to binary values
| mode | str, optional | Mode to employ at edges. Defaults to 'reflect'.
| alpha_mu | float, optional | Expoenent for mean bias term. Defaults to None.
| alpha_sigma | float, optional | Expoenent for standard deviation bias term. Defaults to None.
| alpha_rho | float, optional | Expoenent for correlation term. Defaults to 1.0.
| max_bias | float, optional | Maximum bias deemed useful. Defaults to None. Will be used to construct exponents alpha_mu and alpha_sigma if they are not specified.
| rho_min | float, optional | Minumum acceptable correlation. Defaults to 0.0.
| C_rho |float, optional | Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.
| C_mu | float, optional | Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.
| C_sigma | float, optional| Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.

Returns:
    float: fssim score, useful threshold
    