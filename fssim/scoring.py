import copy
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr


def get_convoluted_fractions(array: np.ndarray, threshold: float, scale: int, mode='constant'):
    """
    Applies a threshold to data and a convolution filter

    Args:
        array (np.ndarray): Array of data
        threshold (float): Rainfall threshold
        scale (int): Width of filter

    Returns:
        np.ndarray: Filtered array
    """
    if len(array.shape) == 2:
        array = np.expand_dims(array, axis=0)
    
    binary_array = (array >= threshold).astype(np.single)
    
    convoluted_array = np.empty(shape=array.shape)
    
    if scale > 1:
        for n in range(array.shape[0]):
            # Performing this in a loop since older versions
            # of uniform_filter do not accept the axes parameter
            
            if mode == 'constant':
                convoluted_array[n,...] = uniform_filter(binary_array[n,...], 
                                                size=scale, 
                                                mode="constant", 
                                                cval=0.0)
            else:
                convoluted_array[n,...] = uniform_filter(binary_array[n,...], 
                                                size=scale, 
                                                mode=mode)
    else:
        convoluted_array = binary_array
            
    return convoluted_array


def calculate_fractions_SSIM(X_o: np.ndarray, X_f: np.ndarray, 
         scale: int, thr: float, mode: str='constant',
         alpha_mu: float=None, alpha_sigma:float=None, alpha_rho:float=1.0,
         max_bias: float=None, rho_min: float=0.0,
         C_rho=1e-16, C_mu=1e-16, C_sigma=1e-16) -> float:
    """
    Calculate Fractions Structural Similarity Index between forecast and observations.


    Args:
        X_o (np.ndarray): Array of shape (s, m, n) containing s samples of the observation field
        X_f (np.ndarray): Array of shape (s, m, n, en) containing an ensemble of c forecast fields, s samples, or (s,m,n) if just one ensemble member (will be converted to the right shape)
        scale (int): Size of neighbourhood around each pixel to calculate the fraction.
        thr (float): Threshold to convert rainfall values to binary values
        mode (str, optional): Mode to employ at edges. Defaults to 'reflect'.
        alpha_mu (float, optional): Expoenent for mean bias term. Defaults to None.
        alpha_sigma (float, optional): Expoenent for standard deviation bias term. Defaults to None.
        alpha_rho (float, optional): Expoenent for correlation term. Defaults to 1.0.
        max_bias (float, optional): Maximum bias deemed useful. Defaults to None. Will be used to construct exponents alpha_mu and alpha_sigma if they are not specified.
        rho_min (float, optional): Minumum acceptable correlation. Defaults to 0.0.
        C_rho (float, optional): Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.
        C_mu (float, optional): Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.
        C_sigma (float, optional): Additive constant to ensure score behaves well for all zeros. Defaults to 1e-16.

    Returns:
        float: fssim score, useful threshold
    """
    f = lambda x: 2*x / (1+x**2)
    useful_threshold = 0.5*(1 + rho_min)
    
    if alpha_mu is None:
        if max_bias is not None:
            # Set exponents based on maximum bias values
        
            alpha_mu = (np.log(useful_threshold) ) / ( np.log( f(1 + max_bias) ) )
        else:
            alpha_mu = 1.0
    if alpha_sigma is None:
        if max_bias is not None:
            # Set exponents based on maximum bias values
            
            alpha_sigma = (np.log(useful_threshold) ) / ( np.log( f(1 + max_bias) ) )
        else:
            alpha_sigma = 1.0

    X_f = X_f.copy()
    X_o = X_o.copy()
    
    fcst_array_shape = X_f.shape
    obs_array_shape = X_o.shape
    
    if len(fcst_array_shape) == len(obs_array_shape):
        assertion = X_f.shape != X_o.shape
        X_f = np.expand_dims(X_f, axis=-1)
    else:
        assertion = X_f.shape[:-1] != X_o.shape
        
    if assertion:
        message = "fcst_array and obs_array must have the same image dimensions"
        raise ValueError(message)
    
    if len(obs_array_shape) == 2:
        X_f = X_f[None, :, :, :]
        X_o = X_o[None, :, :]

    S_o = get_convoluted_fractions(X_o, threshold=thr, scale=scale, mode=mode)
    
    S_f = np.empty(X_f.shape)
    for ii in range(X_f.shape[-1]):
        S_f[...,ii] = get_convoluted_fractions(X_f[..., ii], 
                                               threshold=thr, 
                                               scale=scale, 
                                               mode=mode)
    
    fcst_mu = S_f.mean()
    fcst_sigma = S_f.std()

    obs_mu = S_o.mean()
    obs_sigma = S_o.std()

    rho = pearsonr(S_f.flatten(), S_o.flatten()).statistic
    
    rho_term = ( 0.5*(1 + rho + C_rho/ (obs_sigma*fcst_sigma)))**alpha_rho
    mu_term = ((2*obs_mu*fcst_mu + C_mu) / (obs_mu**2 + fcst_mu**2))**alpha_mu
    sigma_term = ((2*obs_sigma*fcst_sigma + C_sigma) / (obs_sigma**2 + fcst_sigma**2))**alpha_sigma

    fssim_score = rho_term*mu_term*sigma_term
    
    return fssim_score, useful_threshold

def calculate_correlation(truth_array, fcst_array, window_range, threshold=0.01):
    (height, width) = truth_array.shape

    corr_t_f = {f't_fn_{orientation}': {} for orientation in ['row', 'col']}
    corr_t_f.update({f'f_tn_{orientation}': {} for orientation in ['row', 'col']})
    corr_t_f.update({f't_tn_{orientation}': {} for orientation in ['row', 'col']})
    corr_t_f.update({f'f_fn_{orientation}': {} for orientation in ['row', 'col']})

    p_t_f = copy.deepcopy(corr_t_f )
    
    truth_integer_array = (truth_array >= threshold).astype(np.single)
    fcst_integer_array = (fcst_array >= threshold).astype(np.single)


    for n in window_range:

        t0 = {'row': [], 'col': []}
        tn = {'row': [], 'col': []}
        f0 = {'row': [], 'col': []}
        fn = {'row': [], 'col': []}
        
        for w in np.arange(0, width-n):

            t0['col'] += list(truth_integer_array[ :, w].flatten())
            tn['col'] += list(truth_integer_array[ :, w+n].flatten())
            
            f0['col'] += list(fcst_integer_array[:,w].flatten())
            fn['col'] += list(fcst_integer_array[:,w+n].flatten())
            
        for h in np.arange(0, height-n):
            
            t0['row'] += list(truth_integer_array[ h, :].flatten())
            tn['row'] += list(truth_integer_array[ h+n, :].flatten())
            
            f0['row'] += list(fcst_integer_array[h, :].flatten())
            fn['row'] += list(fcst_integer_array[h+ n, :].flatten())
            
        for orientation in ['row', 'col']:
            corr_t_f[f't_tn_{orientation}'][n] = pearsonr(t0[orientation], tn[orientation]).statistic
            corr_t_f[f'f_fn_{orientation}'][n] = pearsonr(f0[orientation], fn[orientation]).statistic
            corr_t_f[f't_fn_{orientation}'][n] = pearsonr(t0[orientation], fn[orientation]).statistic
            corr_t_f[f'f_tn_{orientation}'][n] = pearsonr(f0[orientation], tn[orientation]).statistic
            p_t_f[f't_fn_{orientation}'][n]= pearsonr(t0[orientation], fn[orientation]).pvalue
            p_t_f[f'f_tn_{orientation}'][n]= pearsonr(f0[orientation], tn[orientation]).pvalue
            p_t_f[f't_tn_{orientation}'][n]= pearsonr(t0[orientation], tn[orientation]).pvalue
            p_t_f[f'f_fn_{orientation}'][n]= pearsonr(f0[orientation], fn[orientation]).pvalue
    return corr_t_f, p_t_f