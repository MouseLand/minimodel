import os
import cv2
import numpy as np
from scipy.io import loadmat

db = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
db[0]['mname'], db[0]['datexp'], db[0]['blk'] = 'L1_A5', '2023_02_27', '1' 
db[1]['mname'], db[1]['datexp'], db[1]['blk'] = 'L1_A1', '2023_03_06', '1'
db[2]['mname'], db[2]['datexp'], db[2]['blk'] = 'FX9', '2023_05_15', '2' 
db[3]['mname'], db[3]['datexp'], db[3]['blk'] = 'FX10', '2023_05_16', '1' 
db[4]['mname'], db[4]['datexp'], db[4]['blk'] = 'FX8', '2023_05_16', '2' 
db[5]['mname'], db[5]['datexp'], db[5]['blk'] = 'FX20', '2023_09_29', '1'


mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20']
exp_date = ['022723', '030623', '051523', '051623', '051623', '092923']
NNs = [6636,6055,3575,4792,5804,2746] # total number of neurons
NNs_valid = [4242,2840,926,3040,2217,1239] # number of neurons with FEV>0.15
img_file_name = ['nat60k_text16_old.mat', # nat60k images and text16 images are original images
                 'nat60k_text16_old.mat',
                 'nat60k_text16.mat',# nat60k images are original images, text16 images are rezscored to match nat60k mean and std
                 'nat60k_text16.mat',
                 'nat60k_text16.mat',
                 'nat60k_text16.mat']

def load_images(root, mouse_id, file='nat60k_text16.mat', downsample=1, normalize=True, crop=True):
    """ 
    load images from mat file.

    Parameters:
    ----------
        root (str): The root directory containing the .mat file.
        mouse_id (int): id of the mouse.
        file (str): The name of the .mat file to load. Default is 'nat60k_text16.mat'.
        downsample (int): Factor by which to downsample the images. Default is 1 (no downsampling).
        normalize (bool): Whether to normalize the images. Default is True.
        crop (bool): Whether to crop the images (only keep the left and center screen). Default is True.

    Returns:
    -------
        img: The preprocessed images with shape (n_images, Ly, Lx).
    """
    path = os.path.join(root, file)
    dstim = loadmat(path, squeeze_me=True) # stimulus data
    img = np.transpose(dstim['img'], (2,0,1)).astype('float32')
    n_stim, Ly, Lx = img.shape
    print('raw image shape: ', img.shape)

    if mouse_id == 5: xrange = [46, 176]
    else: xrange = [0, 130]

    img = np.array([cv2.resize(im, (int(Lx//downsample), int(Ly//downsample))) for im in img])
    if crop:
        img = img[:,:,:int(176//downsample)] # keep left and middle screen
        xrange = [int(xrange[0]//downsample), int(xrange[1]//downsample)]
        img = img[:,:,xrange[0]:xrange[1]] # crop image based on RF locations
        print('cropped image shape: ', img.shape)
    if normalize:
        img -= img.mean()
        img /= img.std()
    print('img: ', img.shape, img.min(), img.max(), img.dtype)
    return img

def load_neurons(file_path, mouse_id = None, fixtrain=False):
    '''
    load neurons of nat60k_text16 recordings.
    file_path: path to the preprocessed file from combine_stim.ipynb file.
    mouse_id: mouse id, used to remove flipped test images for mouse 1,2,3 (2,3 are the l1a2 and l1a3). 
    fixtrain: if True, only keep nat30k images for training.

    Load neurons of nat60k_text16 recordings.

    Parameters:
    ----------
    file_path: str
        Path to the preprocessed file from combine_stim.ipynb file.
    mouse_id: int, optional
        Mouse ID, used to remove flipped test images for mouse 1. (optional, since the including of flipped test images doesn't affect the results too much)
    fixtrain: bool, optional
        If True, only keep nat30k unique images for training.

    Returns:
    -------
    spks : numpy.ndarray, shape (n_stimuli, n_neurons)
        Activities of neurons.
    istim_sp : numpy.ndarray, shape (n_train_stimuli,)
        Stimulus indices for training responses.
    istim_ss : numpy.ndarray, shape (n_test_stimuli,)
        Stimulus indices for testing responses.
    xpos : numpy.ndarray, shape (n_neurons,)
        X cortical positions of neurons.
    ypos : numpy.ndarray, shape (n_neurons,)
        Y cortical positions of neurons.
    spks_rep_all : numpy.ndarray, shape (n_test_stimuli, n_repeats, n_neurons)
        test activities.
    '''
    print(f'\nloading activities from {file_path}')
    dat = np.load(file_path, allow_pickle=True) 
    spks_rep_all = dat['ss_all'] # test responses, 500 x (nrepeats, NN) normally 10 repeats of 500 stim
    ypos, xpos = dat['ypos'], dat['xpos']
    spks = dat['sp']
    istim_sp = (dat['istim_sp']).astype('int')
    istim_ss = (dat['istim_ss']).astype('int')

    if fixtrain:
        idx = np.where(istim_sp<30000)[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    elif mouse_id in [1]: # remove flipped test images for mouse 1
        idx = np.where((istim_sp<30000) | (istim_sp>30500))[0]
        istim_sp = istim_sp[idx]
        spks = spks[:,idx]
    spks = spks.T
    return spks, istim_sp, istim_ss, xpos, ypos, spks_rep_all

def split_train_val(istim_train, train_frac=0.9):
    '''
    split training and validation set.
    train_frac: fraction of training set, 1 - train_frac = val_frac.
    Split training and validation set.

    Parameters:
    ----------
    istim_train : numpy.ndarray
        Array of stimulus indices for the training set.
    train_frac : float, optional
        Fraction of the data to be used for training. The validation 
        fraction will be 1 - train_frac. Default is 0.9.

    Returns:
    -------
    itrain : numpy.ndarray, shape (n_train,)
        Indices for the training set.
    ival : numpy.ndarray, shape (n_val,)
        Indices for the validation set.
    '''
    print('\nsplitting training and validation set...')
    np.random.seed(0)
    itrain = np.arange(len(istim_train))
    val_interval = int(1/(1-train_frac))
    ival = itrain[::val_interval]
    itrain = np.ones(len(itrain), 'bool')
    itrain[ival] = False
    itrain = np.nonzero(itrain)[0]

    print('itrain: ', itrain.shape)
    print('ival: ', ival.shape)
    return itrain, ival

def normalize_spks(spks, spks_rep, itrain):
    '''
    Normalize neuron activities by standard deviation.

    Parameters:
    ----------
    spks : numpy.ndarray, shape (n_stim_train, n_neurons)
        Spiking activities for the training stimuli.
    spks_rep : numpy.ndarray, shape (n_stim_test, nrepeats, n_neurons)
        Repeated spiking activities for the test stimuli.
    itrain : numpy.ndarray, shape (n_stim_train,)
        Indices for the training set.

    Returns:
    -------
    spks : numpy.ndarray, shape (n_stim_train, n_neurons)
        Normalized spiking activities for the training stimuli.
    spks_rep : numpy.ndarray, shape (n_stim_test, nrepeats, n_neurons)
        Normalized repeated spiking activities for the test stimuli.
    '''
    print('\nnormalizing neural data...')
    spks_std = spks[itrain].std(0)
    spks_std[spks_std < 0.01] = 0.01
    spks = spks / spks_std
    for i in range(len(spks_rep)):
        spks_rep[i] /= spks_std
    print('finished')
    return spks, spks_rep
