import numpy as np
import torch
from torch.nn.functional import relu

def eval_gabors(img, gabor_filters, resp, batch_size=100, device=torch.device('cuda'), rectify=True): 
    # response to images
    for k in np.arange(0, img.shape[-1], batch_size):
        kend = min(img.shape[-1]*1.0, 1.0*(k+batch_size))
        kend = int(kend)
        img_batch = torch.from_numpy(img[:,:,k:kend].transpose(2,0,1)).to(device)
        resp[:, k:kend] = (gabor_filters * img_batch 
                                ).sum(axis=(-2,-1)).reshape(resp.shape[0], -1)
    # rectify
    if rectify:
        resp = relu(resp)


def gabor_filter(x, y, xcent, ycent, A, sigma, f, theta, ph, ar, is_torch=True):
    if is_torch:
        cos = torch.cos
        sin = torch.sin
        exp = torch.exp
    else:
        cos = np.cos
        sin = np.sin
        exp = np.exp
    yc = y - ycent
    xc = x - xcent
    cosine = cos(ph + f * (yc * cos(theta) + xc * sin(theta)))
    gaussian = exp(-(xc**2/ar + yc**2) / (2*sigma**2))
    G = A * gaussian * cosine
    return G

def fit_gabor_model(X, img, X_test, img_test, X_test_real=None, device = torch.device('cuda')):
    '''
    fit gabor RFs to neuron responses.
    X: n_stim x neurons
    img: Ly x Ly x n_stim, should be zscored across stimuli already.
    '''
    Ly, Lx, n_stim = img.shape
        
    n_train, n_neurons = X.shape

    # zscore X
    # X = (X - X.mean(axis=0)) / X.std(axis=0) # zscore X across stimuli
    X_train = torch.from_numpy(X.astype('float32'))

    train_mu = X_train.mean(dim=0)
    train_std = X_train.std(dim=0)
    X_train = (X_train - train_mu) / train_std

    # define gabor parameters
    sigma = np.array([0.75, 1.25, 1.5, 2.5, 3.5, 4.5, 5.5])
    f = np.array([0.1, 0.25, 0.5, 1, 2]) #[.01:.02:.13];
    theta = np.arange(0, np.pi, np.pi/8)
    ph = np.arange(0, 2*np.pi, np.pi/4)
    ar = np.array([1, 1.5, 2])
    print(f'sigma: {sigma.shape}, f: {f.shape}, theta: {theta.shape}, ph: {ph.shape}, ar: {ar.shape}')

    params = np.meshgrid(sigma, f, theta, ph, ar, indexing='ij')
    n_gabors = params[0].size
    print(f'number of gabors: {n_gabors}')

    for i in range(len(params)):
        params[i] = np.expand_dims(params[i], axis=(-2,-1))
        params[i] = torch.from_numpy(params[i].astype('float32'))
    sigma, f, theta, ph, ar = params
    print(f'sigma: {sigma.shape}, f: {f.shape}, theta: {theta.shape}, ph: {ph.shape}, ar: {ar.shape}')

    # gabor locations
    ys, xs = np.meshgrid(np.arange(0,Ly), np.arange(0,Lx), indexing='ij')
    ys, xs = torch.from_numpy(ys.astype('float32')), torch.from_numpy(xs.astype('float32'))

    # store best parameters
    vmax = -np.inf * np.ones((n_neurons,), 'float32')
    Amax = -np.inf * np.ones((n_neurons,2), 'float32')
    vmax_test = -np.inf * np.ones((n_neurons,), 'float32')
    imax = np.zeros((n_neurons,), 'int')
    gmax = np.zeros((n_neurons,), 'int')
    ymax = np.zeros((n_neurons,), 'int')
    xmax = np.zeros((n_neurons,), 'int')
    mu1 = -np.inf * np.ones((n_neurons,), 'float32')
    mu2 = -np.inf * np.ones((n_neurons,), 'float32')
    mu1 = torch.from_numpy(mu1).to(device)
    mu2 = torch.from_numpy(mu2).to(device)


    resp_train1 = torch.zeros((n_gabors, n_train), dtype=torch.float32, device=device)
    resp_train2 = torch.zeros((n_gabors, n_train), dtype=torch.float32, device=device)
    X_train = X_train.to(device)

    ycents, xcents = np.meshgrid(np.arange(0,Ly-1,1), np.arange(1,Lx,1), indexing='ij')
    ycents, xcents = torch.from_numpy(ycents.astype('float32')), torch.from_numpy(xcents.astype('float32'))

    vtest = np.zeros(ycents.shape, 'float32')
    print(f'ycents: {ycents.shape}, xcents: {xcents.shape}')

    # from .gabor import gabor_filter, eval_gabors
    import time
    from torch.nn.functional import relu

    tic = time.time()

    for ycent, xcent in zip(ycents.flatten(), xcents.flatten()):
        # compute responses to gabors
        gabor_filters1 = gabor_filter(ys, xs, ycent, xcent, 1, sigma, f, theta, ph, ar, 
                                        is_torch=True).to(device).unsqueeze(-3)
        eval_gabors(img, gabor_filters1, resp_train1, device=device, rectify=False) 

        # compute responses to gabors with phase shifted by pi/2
        gabor_filters2 = gabor_filter(ys, xs, ycent, xcent, 1, sigma, f, theta, ph + np.pi/2, ar, 
                                        is_torch=True).to(device).unsqueeze(-3) # phase shifted by pi/2
        eval_gabors(img, gabor_filters2, resp_train2, device=device, rectify=False)

        # compute complex cell responses
        resp_train2 = torch.sqrt(resp_train1**2 + resp_train2**2) # RMS for complex cell response
        # rectify
        resp_train2 = relu(resp_train2) # (n_gabors, n_stim)
        resp_train1 = relu(resp_train1) # (n_gabors, n_stim)
        
        mu1_train = resp_train1.mean(axis=-1)
        resp_train1 -= mu1_train.unsqueeze(-1)
        mu2_train = resp_train2.mean(axis=-1)
        resp_train2 -= mu2_train.unsqueeze(-1)

        # compute coefficients for best fit: r = c1*resp1 + c2*resp2 
        # r = X @ c
        # c = (X^T X)^-1 X^T r
        g11 = (resp_train1**2).sum(axis=-1) # variance， size: (n_gabors,)
        g12 = (resp_train1 * resp_train2).sum(axis=-1) # covariance， size: (n_gabors,)
        g22 = (resp_train2**2).sum(axis=-1) # variance， size: (n_gabors,)

        # Compute determinant of the covariance matrix
        det = g11 * g22 - g12**2  # Determinant, shape: (n_gabors,)

        # Compute the inverse of the covariance matrix using the adjoint method
        idet = 1.0 / (det + 1e-6)  # Add small regularization to avoid division by zero

        # Compute coefficients
        # X^T * r
        Xtr1 = (resp_train1 @ X_train)  # Shape: (n_gabors, n_neurons)
        Xtr2 = (resp_train2 @ X_train)  # Shape: (n_gabors, n_neurons)

        # c1 (coefficient for simple cell)
        c1 = idet[:, None] * (g22[:, None] * Xtr1 - g12[:, None] * Xtr2)  # Shape: (n_gabors, n_neurons)

        # c2 (coefficient for complex cell)
        c2 = idet[:, None] * (-g12[:, None] * Xtr1 + g11[:, None] * Xtr2)  # Shape: (n_gabors, n_neurons)

        # Rectify coefficients (ensure non-negativity)
        c1 = relu(c1)
        c2 = relu(c2)

        cc = c1 * Xtr1 + c2 * Xtr2  # correlation, shape: (n_gabors, n_neurons)
        
        vbest, gbest = cc.max(axis=0) # return max value and index
        vbest, gbest = vbest.cpu().numpy(), gbest.cpu().numpy()
        c = torch.stack((c1, c2), dim=1) # size: (n_gabors, 2, n_neurons)
        c = c[gbest, :, np.arange(n_neurons)] # size: (n_neurons, 2)

        rpred = (resp_train1[gbest].T - resp_train1[gbest].mean(axis=-1)) * c[:,0] + (resp_train2[gbest].T - resp_train2[gbest].mean(axis=-1)) * c[:,1] # (n_stim, n_neurons)
        vexp = 1 - ((rpred - X_train)**2).mean(axis=0) / (X_train**2).mean(axis=0)
        vexp = vexp.cpu().numpy()

        imax = vmax < vexp # update only if new value is greater than old value for each neuron
        vmax[imax] = vexp[imax]
        gmax[imax] = gbest[imax]    
        Amax[imax] = c[imax].cpu().numpy()

        ymax[imax] = ycent.numpy()
        xmax[imax] = xcent.numpy()

        mu1[imax] = mu1_train[gbest][imax]
        mu2[imax] = mu2_train[gbest][imax]

        if xcent.numpy() % 5 == 0:                                                                                                                                                                                                
            print(f'y={ycent.numpy():.0f}, x={xcent.numpy():.0f}, vmax={vmax.mean():.3f}, time {time.time()-tic:.1f}s')

        # test
        # test
    ym = torch.from_numpy(ymax.astype('float32')).unsqueeze(-1).unsqueeze(-1)
    xm = torch.from_numpy(xmax.astype('float32')).unsqueeze(-1).unsqueeze(-1)
    # print(f'ym: {ym.shape}, xm: {xm.shape}')
    gabor_params = torch.zeros((5, n_neurons, 1, 1))
    for i in range(len(gabor_params)):
        gabor_params[i] = params[i].flatten()[gmax].reshape(n_neurons, 1, 1)
    msigma, mf, mtheta, mph, mar = gabor_params
    gabor_filters1 = gabor_filter(ys, xs, ym, xm, 1, msigma, mf, mtheta, mph, mar, is_torch=True).to(device).unsqueeze(-3)
    gabor_filters2 = gabor_filter(ys, xs, ym, xm, 1, msigma, mf, mtheta, mph + np.pi/2, mar, is_torch=True).to(device).unsqueeze(-3)

    # load test images
    # img_test = img_all[istim_test].transpose(1,2,0)
    # img_test = (img_test - img_mean) / img_std
    print(f'img_test: {img_test.shape} {img_test.min()}, {img_test.max()}')

    # predict responses
    ntest = img_test.shape[-1]
    resp_test1 = torch.zeros((n_neurons, ntest), dtype=torch.float32, device=device)
    resp_test2 = torch.zeros((n_neurons, ntest), dtype=torch.float32, device=device)
    eval_gabors(img_test, gabor_filters1, resp_test1, device=device, rectify=False)
    eval_gabors(img_test, gabor_filters2, resp_test2, device=device, rectify=False)
    resp_test2 = torch.sqrt(resp_test1**2 + resp_test2**2) # RMS for complex cell response
    resp_test2 = relu(resp_test2) # rectify
    resp_test1 = relu(resp_test1) # rectify
    c = torch.from_numpy(Amax).to(device)

    rpred = ((resp_test1.T - mu1) * c[:,0] + (resp_test2.T - mu2) * c[:,1]) # (n_stim, n_neurons)
    print(f'rpred: {rpred.shape}')

    # test responses
    spks_rep_all = X_test.copy()
    nreps = []
    for i in range(len(spks_rep_all)):
        spks_rep_all[i] -= train_mu.cpu().numpy()
        spks_rep_all[i] /= train_std.cpu().numpy()
        nreps.append(spks_rep_all[i].shape[0])

    from . import metrics
    fev, feve = metrics.monkey_feve(spks_rep_all, rpred.cpu().numpy(), nreps)
    print(f'fev:{fev.mean():.3f}, feve:{feve.mean():.3f}')

    result_dict = {'msigma': msigma.cpu().numpy().squeeze(), 'mf': mf.cpu().numpy().squeeze(), 'mtheta': mtheta.cpu().numpy().squeeze(), 'mph': mph.cpu().numpy().squeeze(), 'mar': mar.cpu().numpy().squeeze(), \
        'Amax': Amax, 'vmax': vmax, 'vmax_test': vmax_test, 'ymax': ymax, 'xmax': xmax, 'mu1': mu1.cpu().numpy(), 'mu2': mu2.cpu().numpy(), "train_mu": train_mu.cpu().numpy(), 'train_std': train_std.cpu().numpy(), \
        'fev': fev, 'feve': feve, 'rpred': rpred.cpu().numpy(), 'gmax': gmax}
    
    return result_dict
            

