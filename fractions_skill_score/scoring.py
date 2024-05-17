import copy
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr


def get_convolved_fractions(array: np.ndarray, threshold: float, scale: int, mode='constant'):
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
    
    convolved_array = np.empty(shape=array.shape)
    
    if scale > 1:
        for n in range(array.shape[0]):
            # Performing this in a loop since older versions
            # of uniform_filter do not accept the axes parameter
            
            if mode == 'constant':
                convolved_array[n,...] = uniform_filter(binary_array[n,...], 
                                                size=scale, 
                                                mode="constant", 
                                                cval=0.0)
            else:
                convolved_array[n,...] = uniform_filter(binary_array[n,...], 
                                                size=scale, 
                                                mode=mode)
    else:
        convolved_array = binary_array
            
    return convolved_array


def fss_equation(mu_obs, mu_f, sigma_obs, sigma_f, rho):
    
    return (2*mu_obs*mu_f + 2 *rho * sigma_f * sigma_obs)/(mu_obs**2 + mu_f**2 + sigma_obs**2 + sigma_f**2)


def check_input_dims(X_o: np.ndarray, X_f: np.ndarray):

    """
    Make dimensions of arrays consistent for calculation of FSS

    Args:
        X_o (np.ndarray): Array of shape (s, m, n) containing s samples of the observation field
        X_f (np.ndarray): Array of shape (s, m, n) containing an ensemble of c forecast fields, s samples, 

    """

    X_f = X_f.copy()
    X_o = X_o.copy()

    if len(X_f.shape) == len(X_o.shape):
        # Add dummy ensemble dimension
        X_f = np.expand_dims(X_f, axis=-1)

    if X_f.shape[:-1] != X_o.shape:
        raise ValueError("fcst_array and obs_array must have the same image dimensions")
    
    if len(X_o.shape) == 2:
        # Add dummy samples dimension if only one sample
        X_f = X_f[None, :, :, :]
        X_o = X_o[None, :, :]

    return X_o, X_f

def get_summary_stats(X_o: np.ndarray, X_f: np.ndarray, 
                      threshold_type: str,
                      scale: int, thr: float, mode: str='reflect'):
    """
    Calculate neighbourhood mean, neighbourhood standard deviation, and neighbourhood correlation 

    Args:
        X_o (np.ndarray): Array of shape (s, m, n) containing s samples of the observation field
        X_f (np.ndarray): Array of shape (s, m, n) containing an ensemble of c forecast fields, s samples, or (s,m,n) if just one ensemble member (will be converted to the right shape)
        threshold_type (str): one of 'percentile' or 'absolute'. If 'percentile' then the threshold is a percentile of the observations or forecast.
        scale (int): Size of neighbourhood around each pixel to calculate the fraction.
        thr (float): Threshold to convert rainfall values to binary values
        mode (str, optional): Mode to employ at edges. Defaults to 'reflect'.
    Returns:
        _type_: _description_
    """

    if threshold_type == 'percentile':
        forecast_threshold = np.percentile(X_f[:,:], thr)
        obs_threshold = np.percentile(X_o[:,:], thr)
    else:
        forecast_threshold = thr
        obs_threshold = thr

    S_o = get_convolved_fractions(X_o, threshold=obs_threshold, scale=scale, mode=mode)
    
    S_f = np.empty(X_f.shape)
    for ii in range(X_f.shape[-1]):
        S_f[...,ii] = get_convolved_fractions(X_f[...,ii], threshold=forecast_threshold, scale=scale, mode=mode)
        
    mu_f = S_f.mean()
    sigma_f = S_f.std()

    mu_obs = S_o.mean()
    sigma_obs = S_o.std()

    rho = pearsonr(S_f.flatten(), S_o.flatten()).statistic

    return mu_f, sigma_f, mu_obs, sigma_obs, rho


def fss(X_o: np.ndarray, X_f: np.ndarray, 
        threshold_type: str,
        scale: int, thr: float,
        mode: str='reflect') -> float:
    """
    Calculate Fractions Structural Similarity Index between forecast and observations.


    Args:
        X_o (np.ndarray): Array of shape (s, m, n) containing s samples of the observation field
        X_f (np.ndarray): Array of shape (s, m, n) containing an ensemble of c forecast fields, s samples, or (s,m,n) if just one ensemble member (will be converted to the right shape)
        threshold_type (str): one of 'percentile' or 'absolute'. If 'percentile' then the threshold is a percentile of the observations or forecast.
        scale (int): Size of neighbourhood around each pixel to calculate the fraction.
        thr (float): Threshold to convert rainfall values to binary values
        mode (str, optional): Mode to employ at edges. Defaults to 'reflect'.
       
    Returns:
        float: fss score
    """


    X_o, X_f = check_input_dims(X_f=X_f, X_o=X_o)
    
    mu_f, sigma_f, mu_obs, sigma_obs, rho = get_summary_stats(X_o=X_o, 
                                                              X_f=X_f, 
                                                              scale=scale, 
                                                              thr=thr, 
                                                              mode=mode, 
                                                              threshold_type=threshold_type)

    fss = fss_equation(mu_obs, mu_f, sigma_obs, sigma_f, rho)
    
    return fss


def fss_random(X_o: np.ndarray,
               threshold_type: str,
               scale: int, thr: float, mode: str='reflect') -> float:
    """
    Calculate Fractions Skill Score for a random Bernoulli forecast

    Args:
        X_o (np.ndarray): Array of shape (s, m, n) containing s samples of the observation field
        threshold_type (str): one of 'percentile' or 'absolute'. If 'percentile' then the threshold is a percentile of the observations or forecast.
        scale (int): Size of neighbourhood around each pixel to calculate the fraction.
        thr (float): Threshold to convert rainfall values to binary values
        mode (str, optional): Mode to employ at edges. Defaults to 'reflect'.
       
    Returns:
        float: fss score for random data
    """


    X_o, _ = check_input_dims(X_f=X_o, X_o=X_o)
    
    _, _, mu_obs, sigma_obs, _ = get_summary_stats(X_o=X_o, 
                                                    X_f=X_o, 
                                                    scale=scale, 
                                                    thr=thr, 
                                                    mode=mode, 
                                                    threshold_type=threshold_type)
    if threshold_type == 'percentile':
        mu_obs_0 = (X_o > np.percentile(X_o,thr)).mean()
    else:
        mu_obs_0 = (X_o > thr).mean()
    
    _, _, mu_obs, sigma_obs, _ = get_summary_stats(X_o=X_o, X_f=X_o, scale=scale, thr=thr, mode=mode, threshold_type=threshold_type)

    sigma_bernoulli_sqd = (1/scale**2)*(mu_obs_0 - mu_obs_0**2)
    sigma_fcst = np.sqrt(sigma_bernoulli_sqd)
    
    fss_random = fss_equation(mu_obs=mu_obs, mu_f=mu_obs, sigma_obs=sigma_obs, sigma_f=sigma_fcst, rho=0)
    
    return fss_random
