import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os
import seaborn as sns
from matplotlib import gridspec

def panela_images(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 0])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Texture images')
    pimg = dat['dataset_imgs']
    ids = dat['dataset_img_ids'] 
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]+0.025, pos[2]-0.00, pos[3]+0])
    xpad = int(18)
    ypad = int(35)
    Ly = 66
    ax.imshow(pimg,cmap='gray')
    for i,idd in enumerate(ids):
        ax.text(-5+i*xpad,-15+i*(ypad)+Ly, 'class %d  '%idd, ha='right', va='center') #, transform=ax.transAxes)
    ax.text(pimg.shape[1]-80, pimg.shape[0]+25, 'x 12800', ha='right', fontweight='bold')
    ax.axis('off')
    return il

def panelb_decoding_accuracy(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 1])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('Texture class decoding')
    ax1.set_position([pos[0] - 0.03, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.0, pos[1]+0.01, pos[2]-0.02, pos[3]-0.0])
    mycmap = sns.color_palette("rocket", as_cmap=True)
    n_mouse = 4
    n_neurons = dat['n_neurons']
    all_accs = dat['classification_accs']

    for i in range(n_mouse):
        mean_acc = all_accs[i].mean(1)
        std_acc = all_accs[i].std(1)
        ax.plot(n_neurons, mean_acc, '-o', color=mycmap((i+2)/6), alpha=0.5, label=f'mouse {i+1+2}')
        ax.text(0.1, 0.9-0.08*i, f'mouse{i+1+2}', transform=ax.transAxes, color=mycmap((i+2)/6))
    ax.axhline(1/16, color='gray', linestyle='--', label='chance')
    ax.text(0.77, 0.12, 'chance', transform=ax.transAxes, color='gray')
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('Decoding accuracy (%)')
    # set xaxis to be log scale
    ax.set_xscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(np.arange(0, 0.61, 0.2), labels=[f'{i*100:.0f}' for i in np.arange(0, 0.61, 0.2)])
    ax.set_ylim([0., 0.6])

    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    return il

def panelc_example_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0] - 0.05, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]-0.02, pos[1]+0.01, pos[2]-0.04, pos[3]-0.0])
    ax1.set_title('High FECV neuron example')
    unique_istims_cls1 = dat['example_istim_cls1']
    unique_istims_cls2 = dat['example_istim_cls2']
    neuron_spks = dat['example_neuron_spks']
    classes = dat['example_classes']
    ineurons = dat['example_ineuron']
    txt16_istim_test = dat['txt16_istim_test']

    for i, istim in enumerate(unique_istims_cls1):
        idxes = np.where(txt16_istim_test==istim)[0]
        ax.scatter(neuron_spks[0, idxes].mean(), neuron_spks[1, idxes].mean(), label=f'class {classes[0]+1}', color='r', s=10)
    for i, istim in enumerate(unique_istims_cls2):
        idxes = np.where(txt16_istim_test==istim)[0]
        ax.scatter(neuron_spks[0, idxes].mean(), neuron_spks[1, idxes].mean(), label=f'class {classes[1]+1}', color='b', s=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Neuron i')
    ax.set_ylabel('Neuron j')
    ax.set_xlim([-0.1, 4])
    ax.set_ylim([-0.1, 2.5])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([1, 2])
    ax.set_aspect(0.9/ax.get_data_ratio())
    ax.text(0.6, 0.8, f'class {classes[0]+1}', color='r', transform=ax.transAxes, ha='left', va='bottom')
    ax.text(0.6, 0.7, f'class {classes[1]+1}', color='b', transform=ax.transAxes, ha='left', va='bottom')
    return il

def paneld_model_neural_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 3])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('Model and neural FECV')
    ax1.set_position([pos[0] - 0.05, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.0, pos[1]+0.01, pos[2]-0.05, pos[3]-0.0])
    x = dat['minimodel_catvar_noise_all']
    y = dat['neural_catvar_all']
    ax.scatter(x, y, alpha=0.5, s=20, c='gray', edgecolors='white', rasterized=True)
    from scipy.stats import pearsonr
    print(pearsonr(x, y))
    r, p = pearsonr(x, y)
    if p < 0.001: pstr = f'p < {0.001}'
    elif p < 0.05: pstr = f'p < {0.05}'
    else: pstr = f'p = {p:.3f}'
    ax.text(0.6, 0.1, f'r = {r:.3f}\n{pstr}', transform=ax.transAxes, fontsize=10)

    ax.set_xlim([-0.01, 0.1])  
    ax.set_ylim([-0.01, 0.1])
    ax.plot([-0.025, 0.1], [-0.025, 0.1], 'k--')
    ax.set_yticks([0, 0.05, 0.1])

    ax.set_ylabel('Model FECV \n(with Poisson noise)')
    ax.set_xlabel('Neural FECV')
    ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return il

def panele_minimodel_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, :2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('Model FECV across stages')
    ax1.set_position([pos[0] - 0.0, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.03, pos[1]+0.015, pos[2]-0.1, pos[3]-0.0])
    rfsize = dat['rfsize']
    # rfsize = np.pi * rfsize * (270/264)*(65/66)
    rfsize = np.sqrt(rfsize* (270/264)*(65/66)) * 2
    # op_all = dat['op_all']
    op_all = ['conv1', 'conv1(ReLU)', 'conv1(pool)', 'conv2(1x1)', 'conv2(spatial)', 'conv2(ReLU)', 'readout(Wxy)', 'readout(Wc+ELU)']
    minimodel_catvar_all = dat['minimodel_catvar_all']
    NN = len(rfsize)
    cmap = plt.cm.get_cmap('plasma')
    vmax = 25 # 600
    vmin = 5
    rfsize_normalized = (rfsize - vmin) / (vmax - vmin)  # Normalize rfsize
    # Create segments for the LineCollection
    segments = [np.column_stack([np.arange(minimodel_catvar_all.shape[1]), minimodel_catvar_all[i]]) for i in range(minimodel_catvar_all.shape[0])]
    colors = [cmap(rfsize_normalized[i]) for i in range(len(rfsize_normalized))]

    # Create a LineCollection from the segments and set the colors
    from matplotlib.collections import LineCollection
    lc = LineCollection(segments, colors=colors, linewidths=1, alpha=0.3, rasterized=True)
    ax.add_collection(lc)
    
    ax.set_xticks(np.arange(len(op_all)), op_all, rotation=30, ha='right', fontsize=10)                                
    ax.set_ylabel('Mean FECV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, alpha=1, pad=0.04, fraction=0.02)
    # cbar.set_label('Pooling area (deg$^2$)')
    cbar.set_label('Pooling diameter (deg)')
    cbar.solids.set(alpha=1)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_fontsize('small')
    cbar.ax.tick_params(labelsize='small')
    ax.set_ylim(-0.03, 0.4)
    ax.set_xlim(-0.2, 7.2)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_aspect(0.4/ax.get_data_ratio())
    ax.set_title('mouse', loc='center', y=1)
    return il

def panelf_rfsize_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('Pooling diameter')
    ax1.set_position([pos[0] - 0.05, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]-0.01, pos[1]+0.01, pos[2]-0.04, pos[3]-0.0])
    x = dat['pred_catvar_all']

    rfsize = dat['rfsize']
    rfsize = np.sqrt(rfsize* (270/264)*(65/66)) * 2 
    from scipy.stats import pearsonr
    r, p = pearsonr(x, rfsize)
    ax.scatter(x, rfsize, alpha=0.5, c='gray', s=20, edgecolors='white', rasterized=True)
    from scipy.interpolate import UnivariateSpline
    y = rfsize
    x = x
    # fit a linear model
    m, b = np.polyfit(x, y, 1)
    x = np.sort(x)
    ax.plot(x, m*x+b, '--', color='k', lw=1)
    # ax.set_ylabel('Pooling area (deg$^2$)')
    ax.set_ylabel('Pooling diameter (deg)')
    ax.set_xlabel('Model FECV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if p < 0.001: pstr = f'p < {0.001}'
    elif p < 0.05: pstr = f'p < {0.05}'
    else: pstr = f'p = {p:.3f}'
    ax.text(0.6, 0.1, f'r = {r:.3f}\n{pstr}', transform=ax.transAxes, fontsize=10)
    ax.set_xlim([-0.01, 0.4])
    # ax.set_ylim([0, 1000])
    ax.set_ylim([0, 40])
    ax.set_yticks([0, 10, 20, 30, 40])
    # make the size of the x and y ticks the same
    ax.set_aspect(0.9/ax.get_data_ratio(), adjustable='box')
    ax.set_title('mouse', loc='center', y=1)
    return il

def panelg_corr_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 3])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('Input diversity')
    ax1.set_position([pos[0] - 0.03, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]-0.0, pos[1]+0.01, pos[2]-0.04, pos[3]-0.0])
    x = dat['pred_catvar_all']
    minimodel_corr_all = dat['minimodel_poscorr_all'] 
    from scipy.stats import pearsonr
    r, p = pearsonr(x, minimodel_corr_all)
    ax.scatter(x, minimodel_corr_all, alpha=0.5, c='gray', s=20, edgecolors='white', rasterized=True)
    ax.set_xlabel('Model FECV')
    ax.set_ylabel('Mean correlation of \npositive channels')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-0.2, 1])
    ax.set_xlim([-0.025, 0.4])
    y = minimodel_corr_all
    x = x
    # fit a linear model
    m, b = np.polyfit(x, y, 1)
    x = np.sort(x)
    ax.plot(x, m*x+b, '--', color='k', lw=1)
    if p < 0.001: pstr = f'p < {0.001}'
    elif p < 0.05: pstr = f'p < {0.05}'
    else: pstr = f'p = {p:.3f}'
    ax.text(0.6, 0.7, f'r = {r:.3f}\n{pstr}', transform=ax.transAxes, fontsize=10)
    # make the size of the x and y ticks the same
    ax.set_aspect(0.9/ax.get_data_ratio(), adjustable='box')
    ax.set_title('mouse', loc='center', y=1)
    return il

def panelh_monkey_minimodel_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, :2])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0] - 0.0, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.03, pos[1]+0.03, pos[2]-0.1, pos[3]-0.0])
    feve_all = dat['monkey_feve']
    idx = np.where(feve_all > 0.25)[0]
    rfsize = dat['monkey_rfsize'][idx]
    rfsize = np.sqrt(rfsize* (1.1/80)**2) * 2
    op_all = ['conv1', 'conv1(ReLU)', 'conv1(pool)', 'conv2(1x1)', 'conv2(spatial)', 'conv2(ReLU)', 'readout(Wxy)', 'readout(Wc+ELU)']
    minimodel_catvar_all = dat['monkey_minimodel_catvar_all'][idx]
    NN = len(rfsize)
    cmap = plt.cm.get_cmap('plasma')
    vmax = 0.6 # 0.15

    for i in range(NN):
        ax.plot(minimodel_catvar_all[i], '-', alpha=0.4, color=cmap((rfsize[i]-0.0)/vmax))

    ax.set_xticks(np.arange(len(op_all)), op_all, rotation=30, ha='right', fontsize=10)                    
    ax.set_ylabel('Mean FECV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, alpha=1, pad=0.04, fraction=0.02)
    cbar.set_label('Pooling diameter (deg)')
    cbar.solids.set(alpha=1)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_fontsize('small')
    cbar.ax.tick_params(labelsize='small')
    ax.set_ylim(-0.03, 0.2)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_xlim(-0.2, 7.2)
    ax.set_aspect(0.4/ax.get_data_ratio())
    ax.set_title('monkey', loc='center', y=1)
    return il

def paneliii_monkey_rfsize_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 2])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0] - 0.05, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]-0.01, pos[1]+0.03, pos[2]-0.04, pos[3]-0.0])
    feve_all = dat['monkey_feve']
    idx = np.where(feve_all > 0.25)[0]
    x = dat['monkey_pred_catvar'][idx]
    

    rfsize = dat['monkey_rfsize'][idx]
    rfsize = np.sqrt(rfsize* (1.1/80)**2) * 2
    from scipy.stats import pearsonr
    r, p = pearsonr(x, rfsize)
    ax.scatter(x, rfsize, alpha=0.5, c='gray', s=20, edgecolors='white', rasterized=True)
    y = rfsize
    x = x
    # fit a linear model
    m, b = np.polyfit(x, y, 1)
    x = np.sort(x)
    ax.plot(x, m*x+b, '--', color='k', lw=1)
    # ax.set_ylabel('Pooling area (deg$^2$)')
    ax.set_ylabel('Pooling diameter (deg)')
    ax.set_xlabel('Model FECV')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if p < 0.001: pstr = f'p < {0.001}'
    elif p < 0.05: pstr = f'p < {0.05}'
    else: pstr = f'p = {p:.3f}'
    ax.text(0.6, 0.1, f'r = {r:.3f}\n{pstr}', transform=ax.transAxes, fontsize=10)
    ax.set_xlim([-0.01, 0.2])
    # ax.set_ylim([0, 0.35])
    ax.set_ylim([0, 1])
    # make the size of the x and y ticks the same
    ax.set_aspect(0.9/ax.get_data_ratio(), adjustable='box')
    ax.set_title('monkey', loc='center', y=1)
    return il

def panelj_monkey_corr_catvar(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 3])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0] - 0.03, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]-0.0, pos[1]+0.03, pos[2]-0.04, pos[3]-0.0])
    feve_all = dat['monkey_feve']
    idx = np.where(feve_all > 0.25)[0]
    x = dat['monkey_pred_catvar'][idx]
    minimodel_corr_all = dat['monkey_minimodel_poscorr_all'][idx]
    from scipy.stats import pearsonr
    r, p = pearsonr(x, minimodel_corr_all)
    ax.scatter(x, minimodel_corr_all, alpha=0.5, c='gray', s=20, edgecolors='white', rasterized=True)
    ax.set_xlabel('Model FECV')
    ax.set_ylabel('Mean correlation of \npositive channels')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([-0.2, 1])
    ax.set_xlim([-0.01, 0.2])
    y = minimodel_corr_all
    x = x
    # fit a linear model
    m, b = np.polyfit(x, y, 1)
    x = np.sort(x)
    ax.plot(x, m*x+b, '--', color='k', lw=1)
    if p < 0.001: pstr = f'p < {0.001}'
    elif p < 0.05: pstr = f'p < {0.05}'
    else: pstr = f'p = {p:.3f}'
    ax.text(0.6, 0.7, f'r = {r:.3f}\n{pstr}', transform=ax.transAxes, fontsize=10)
    # make the size of the x and y ticks the same
    ax.set_aspect(0.9/ax.get_data_ratio(), adjustable='box')
    ax.set_title('monkey', loc='center', y=1)
    return il

def figure4(dat, root):
    fig = plt.figure(figsize=(14,9))
    grid = plt.GridSpec(3,4, figure=fig, left=0.06, right=0.98, top=0.96, bottom=0.02, 
                        wspace = 0.35, hspace = 0.2, height_ratios=[1, 1, 1])
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_images(fig, grid, dat, il, transl)
    il = panelb_decoding_accuracy(fig, grid, dat, il, transl)
    il = panelc_example_catvar(fig, grid, dat, il, transl)
    il = paneld_model_neural_catvar(fig, grid, dat, il, transl)
    il = panele_minimodel_catvar(fig, grid, dat, il, transl)
    il = panelf_rfsize_catvar(fig, grid, dat, il, transl)
    il = panelg_corr_catvar(fig, grid, dat, il, transl)
    il = panelh_monkey_minimodel_catvar(fig, grid, dat, il, transl)
    il = paneliii_monkey_rfsize_catvar(fig, grid, dat, il, transl)
    il = panelj_monkey_corr_catvar(fig, grid, dat, il, transl)
    fig.savefig(os.path.join(root, 'figure4.pdf'), dpi=200) 