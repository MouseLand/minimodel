
import numpy as np
from scipy.stats import zscore

def fev(spks):
    """
    Calculate the fraction of explainable variance (FEV) for each neuron.

    Parameters:
    ----------
    spks : list of numpy.ndarray
        List of n_images arrays, each of shape (n_repeats, n_neurons), where 
        different images might have different numbers of repeats.

    Returns:
    -------
    fev : numpy.ndarray, shape (n_neurons,)
        Fraction of explainable variance for each neuron.
    """
    n_images = len(spks)

    # calculate total variance of each neuron
    total_var = np.var(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    noise_var = []
    for i in range(n_images):
        noise_var.append(np.var(spks[i], axis=0, ddof=1))
    noise_var = np.vstack(noise_var).mean(axis=0) # shape (n_neurons,)
    
    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    return fev

def feve(spks, spks_pred, multi_repeats=True):
    """
    Calculate the fraction of explainable variance (FEV) and the fraction of explainable variance explained (FEVE) for each neuron.

    Parameters:
    ----------
    spks : list of numpy.ndarray
        List of n_images arrays, each of shape (n_repeats, n_neurons), where 
        different images might have different numbers of repeats.
    spks_pred : numpy.ndarray
        Array of predicted activities, of shape (n_images, n_neurons).
    multi_repeats : bool, optional
        If True, account for multiple repeats in the noise variance calculation. Default is True.

    Returns:
    -------
    fev : numpy.ndarray, shape (n_neurons,)
        Fraction of explainable variance for each neuron.
    feve : numpy.ndarray, shape (n_neurons,)
        Fraction of explainable variance explained for each neuron.
    """
    n_images = len(spks)

    # calculate total variance of each neuron
    total_var = np.var(np.vstack(spks), axis=0, ddof=1) # shape (n_neurons,)

    # calculate noise variance and variance explained of each neuron
    mse = []
    noise_var = []
    for i in range(n_images):
        mse.append((spks[i] - spks_pred[i])**2)
        noise_var.append(np.var(spks[i], axis=0, ddof=1))
    mse = np.vstack(mse).mean(axis=0) # shape (n_neurons,)
    noise_var = np.vstack(noise_var).mean(axis=0) # shape (n_neurons,)

    if not multi_repeats: noise_var = 0
    
    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    # calculate explainable variance explained of each neuron
    feve = 1 - (mse - noise_var)/ (total_var - noise_var) # shape (n_neurons,)

    return fev, feve

def monkey_feve(spks, spks_pred, repetitions):
    """
    Calculate the fraction of explainable variance (FEV) and the fraction of explainable variance explained (FEVE) for each neuron in monkey data.

    Parameters:
    ----------
    spks : numpy.ndarray
        Array of activities, of shape (n_repeats, n_images, n_neurons).
    spks_pred : numpy.ndarray
        Array of predicted activities, of shape (n_images, n_neurons).
    repetitions : list of int
        List of integers indicating the number of repetitions for each neuron.

    Returns:
    -------
    fev : numpy.ndarray, shape (n_neurons,)
        Fraction of explainable variance for each neuron.
    feve : numpy.ndarray, shape (n_neurons,)
        Fraction of explainable variance explained for each neuron.
    """

    n_neurons = spks.shape[-1]

    # calculate total variance, noise variance and variance explained of each neuron
    mse = []
    noise_var = []
    total_var = []
    for i in range(n_neurons):
        mse.append(np.nanmean((spks[:repetitions[i], :, i] - spks_pred[:, i])**2)) 
        noise_var.append(np.nanmean(np.nanvar(spks[:repetitions[i], :, i], axis=0, ddof=1))) 
        total_var.append(np.nanvar(spks[:repetitions[i], :, i], ddof=1))
    mse = np.array(mse)
    noise_var = np.array(noise_var) # shape (n_neurons,)
    total_var = np.array(total_var)

    # calculate explainable variance of each neuron
    fev = (total_var - noise_var) / total_var # shape (n_neurons,)

    # calculate explainable variance explained of each neuron
    feve = 1 - (mse - noise_var)/ (total_var - noise_var) # shape (n_neurons,)

    return fev, feve

def fecv(spks):
    '''
    Calculate the fraction of explainable category variance (FECV).

    Parameters:
    ----------
    spks : numpy.ndarray
        Array of activities, of shape (n_category, n_stim).

    Returns:
    -------
    fecv : float
        Fraction of the total variance that is explainable by the category variance.
    
    '''
    ncat, nstim = spks.shape
    category_mean = spks.mean(axis=1)
    residual_var = np.zeros(ncat)
    for i in range(ncat):
        residual_var[i] = np.sum((spks[i] - category_mean[i]) ** 2) / (nstim - 1)
    residual_var = np.mean(residual_var)
    total_variance = spks.var(ddof=1)
    fecv = (total_variance - residual_var) / total_variance
    return fecv

def fecv_pairwise(spks, labels, ss=None):
    '''
    Calculate the fraction of explainable category variance (FECV) for each pair of categories, assuming the same number of stimuli in each category.

    Parameters:
    ----------
    spks : numpy.ndarray
        Array of activities, of shape (n_neuron, n_stim).
    labels : numpy.ndarray
        Array of labels corresponding to the categories of each stimulus, of shape (n_stim,).
    ss : list, optional
        List of category labels. If None, unique labels from the `labels` array are used.

    Returns:
    -------
    catvar_all : numpy.ndarray
        Mean FECV across all category pairs, of shape (n_neurons)
    '''
    if ss is not None:
        cats = ss
    else:
        cats = np.unique(labels)
    ncat = len(cats)

    nneuron, nstim = spks.shape
    #  zscore the spikes
    spks = zscore(spks, axis=1)

    category_mean = np.zeros((nneuron, ncat))
    for icat, cat in enumerate(cats):
        category_mean[:, icat] = spks[:, labels == cat].mean(axis=1)

    catvar_all = []
    for icat1, cat1 in enumerate(cats):
        for icat2, cat2 in enumerate(cats):
            if icat1 > icat2:
                spks1 = spks[:, labels == cat1]
                spks2 = spks[:, labels == cat2]
                nstim1 = spks1.shape[1]
                nstim2 = spks2.shape[1]
                total_variance = np.var(np.hstack([spks1, spks2]), axis=1, ddof=1)
                residual_var1 = np.sum((spks1 - category_mean[:, icat1][:, None]) ** 2, axis=1) / (nstim1 - 1)
                residual_var2 = np.sum((spks2 - category_mean[:, icat2][:, None]) ** 2, axis=1) / (nstim2 - 1)
                residual_var = np.mean(np.stack([residual_var1, residual_var2]), axis=0)
                category_var = total_variance - residual_var
                category_var[np.abs(category_var) < 0.0001] = 0 # control precision
                # Avoid division by zero by checking if total_variance is zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    category_var = np.where(total_variance != 0, category_var / total_variance, np.nan)
                # print(f'cat1={cat1}, cat2={cat2}, category_var={category_var}, total_variance={total_variance}, residual_var={residual_var}')
                catvar_all.append(category_var)
    catvar_all = np.nanmean(np.stack(catvar_all), axis=0)
    return catvar_all


def add_poisson_noise(predictions, fev_target, init_lam=0.7, idxes=None, N_repeats=10, delta=0.01, return_lam=False):
    '''
    Add Poisson noise to predictions to match the noise level (FEV) in the neurla data.

    Parameters:
    ----------
    predictions : numpy.ndarray
        Array of predicted activities, of shape (N_trials, N_neurons).
    fev_target : float
        The target FEV.
    init_lam : float, optional
        The initial value of noise intensity. Default is 0.7.
    idxes : numpy.ndarray, optional
        The indices of neurons to be added noise. If None, noise is added to all neurons.
    N_repeats : int, optional
        Number of repeats for generating Poisson noise. Default is 10.
    delta : float, optional
        Increment value for adjusting the noise intensity. Default is 0.01.
    return_lam : bool, optional
        If True, also return the final value of lambda. Default is False.

    Returns:
    -------
    noisy_predictions : numpy.ndarray
        Array of noisy predictions, of shape (N_repeats, N_trials, N_neurons).
    noisy_fev : numpy.ndarray
        Array of FEV values for the noisy predictions, of shape (N_neurons,).
    lam : float, optional
        Final value of lambda. Returned only if `return_lam` is True.
    '''
    if idxes is None:
        idxes = np.arange(predictions.shape[-1])
    predictions = predictions[:, idxes]
    lam = init_lam
    stop = False
    track_lam = []
    track_fev = []
    NT, NN = predictions.shape
    # pnoise = np.random.poisson(size=(N_repeats, NT, NN))
    print(f'target fev={fev_target:.3f}')

    while not stop:
        noisy_predictions = predictions[np.newaxis].repeat(N_repeats, axis=0) + np.random.poisson(lam, size=(N_repeats, NT, NN))
        noisy_fev = fev(noisy_predictions.transpose(1, 0, 2))
        mean_noisy_fev = noisy_fev.mean()
        print(f'lam={lam:.3f}, mean_noisy_fev={mean_noisy_fev:.3f}')
        if (np.abs(mean_noisy_fev - fev_target) < 0.001):
            stop = True
        elif lam in track_lam:
            max_idx = np.argmin(np.abs(np.array(track_fev) - fev_target))
            lam = track_lam[max_idx]
            noisy_predictions = predictions[np.newaxis].repeat(N_repeats, axis=0) + np.random.poisson(lam, size=(N_repeats, NT, NN))
            noisy_fev = fev(noisy_predictions.transpose(1, 0, 2))
            mean_noisy_fev = noisy_fev.mean()
            stop = True
        elif mean_noisy_fev > fev_target:
            track_lam.append(lam)
            track_fev.append(mean_noisy_fev)
            lam += delta
        else:
            track_lam.append(lam)
            track_fev.append(mean_noisy_fev)
            lam -= delta
    print(f'final lam={lam:.3f}, mean_noisy_fev={mean_noisy_fev:.3f}')
    if return_lam:
        return noisy_predictions, noisy_fev, lam
    return noisy_predictions, noisy_fev