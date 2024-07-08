import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os
import seaborn as sns
from matplotlib import gridspec

def panela_experiment(fig, grid, dat, il, transl):
    ax = plt.subplot(grid[0, 0])
    ax.axis('off')
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title('Experiment design')
    return il

def panelb_model(fig, grid, dat, il, transl):
    ax = plt.subplot(grid[0, 1])
    ax.axis('off')
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title('Model architecture')
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.00, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    return il

def panelc_activity(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Example neural activity\n(test images)', y=0.91)
    fev = dat['fev_example']
    feve = dat['feve_example']
    spks_test_mean = dat['spks_test_mean_example']
    spks_pred_test = dat['spks_pred_test_example']

    nn_show = 5
    mycmap = plt.cm.get_cmap('tab20c')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.0125, pos[1]-0.0, pos[2]+0.0, pos[3]+0])
    ax = fig.add_axes([pos[0]-0.0125, pos[1]-0.01, pos[2]+0.02, pos[3]+0])
    mycmap = plt.cm.get_cmap('tab20c')

    threshold = 0.15
    np.random.seed(0)
    ineurons_show = np.where((fev>threshold)&(feve>0.9))[0]
    ineurons_show1 = np.random.choice(ineurons_show, 2, replace=False)
    ineurons_show = np.where((fev>threshold)&(feve>0.75)&(feve<0.8))[0]
    ineurons_show2 = np.random.choice(ineurons_show, 2, replace=False)
    ineurons_show = np.where((fev>threshold)&(feve>0.5)&(feve<0.6))[0]
    ineurons_show3 = np.random.choice(ineurons_show, 1, replace=False)
    ineurons_show = np.concatenate([ineurons_show1, ineurons_show2, ineurons_show3])
    isort = np.argsort(feve[ineurons_show])
    # pos = ax.get_position().bounds
    # ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]+0, pos[3]+0])
    for i in range(nn_show):
        ineuron = ineurons_show[isort[i]]
        neuron_feve = feve[ineuron]
        tmp_neural = spks_test_mean[50:200, ineuron]
        tmp_neural = (tmp_neural - tmp_neural.min()) / (tmp_neural.max() - tmp_neural.min())
        tmp_model = spks_pred_test[50:200, ineuron]
        tmp_model = (tmp_model - tmp_model.min()) / (tmp_model.max() - tmp_model.min())
        tmp_neural /= 1.5
        tmp_model /= 1.5
        if i==0:
            ax.plot(tmp_neural + i+0.1*i, label='true', c='black', linewidth=1)
            ax.plot(tmp_model + i+0.1*i, label='predicted', c=mycmap(0), linewidth=1)
        else:
            ax.plot(tmp_neural + i+0.1*i, c='black', label='_nolegend_', linewidth=1)
            ax.plot(tmp_model + i+0.1*i, c=mycmap(0), label='_nolegend_', linewidth=1)
        ax.text(0, i+0.1*i+0.85, f'Neuron {5-i}, FEVE={neuron_feve:.2f}', fontsize=8, va='center')
    # set all axis off
    ax.axis('off')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, nn_show+1)
    # set xticks off
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(1.025, 1.06))
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    return il


def paneld_feve_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE distribution')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.01, pos[2]-0.02, pos[3]-0.0])

    feve_all_mice = dat['feve_all_mice']
    ax.hist(feve_all_mice, bins=20, color='dimgray', alpha=1, edgecolor='white', linewidth=1)
    ax.scatter(np.mean(feve_all_mice), 3300, s=100, c='dimgray', marker='v', label='mean')
    # add the value of mean above the mean
    ax.text(np.mean(feve_all_mice), 3550, f'{np.mean(feve_all_mice):.2f}', ha='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4000)
    ax.set_xlabel('FEVE')
    ax.set_ylabel('# of neurons')
    ax.set_xticks(np.arange(0, 1.01, 0.25), [f'{i:.2f}' for i in np.arange(0, 1.01, 0.25)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panele_5k_30k(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.01, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.01, pos[2]-0.061, pos[3]-0.0])
    # plot the mean of all
    all_feve = dat['5k_30k_feve']
    ax.plot([0, 1], all_feve.mean(axis=0), '-', color='k')
    from scipy.stats import sem
    ax.errorbar([0, 1], all_feve.mean(axis=0), yerr=sem(all_feve, 0), fmt='o', color='k', capsize=5, markersize=3)
    # add scatter plot with lines connecting each pair of points
    nmouse = 6
    for i in range(nmouse):
        ax.scatter([0, 1], all_feve[i], color='gray', alpha=0.3)
        ax.plot([0, 1], all_feve[i], color='gray', alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_ylim(-.49, 1.5)
    ax.set_xlabel('# of train images')
    ax.set_ylabel('FEVE')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.8)
    ax.set_xticks([0, 1], ['5000', '30000'])
    ax.set_yticks(np.arange(0, 0.81, 0.2), [f'{i:.2f}' for i in np.arange(0, 0.81, 0.2)])
    ax.set_aspect(1.5 / ax.get_data_ratio())
    ax.text(1, 0.1, "N=6", transform=ax.transAxes, ha="right", fontsize=14)
    return il

def panelf_depth(fig, grid, dat, il, transl):
    feve_our_model = dat['feve_our_model']
    feve_lurz_model = dat['feve_lurz_model']
    ax1 = plt.subplot(grid[1, 2])
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.03, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('FEVE change with model depth')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.axis('off')
    ax = fig.add_axes([pos[0]-0.002, pos[1]+0.01, pos[2]+0.01, pos[3]-0.0])
    nmouse = 6
    feve_our_model = dat['feve_our_model']
    feve_lurz_model = dat['feve_lurz_model']
    feve_LN = dat['feve_LN_model'].mean()
    from scipy.stats import sem
    cmap = sns.color_palette('rocket')
    c2 = '#4cb938'
    ax.errorbar(np.arange(4), feve_our_model.mean(0), yerr=sem(feve_our_model, 0), c='black', capsize=5, label='our model')
    ax.errorbar(np.arange(4), feve_lurz_model .mean(0), yerr=sem(feve_lurz_model, 0), c=c2, capsize=5, label='sensorium model (2022)', alpha=0.6)
    ax.plot(np.arange(4), feve_our_model.mean(0), 'o-', c='black', label='our model', markersize=3)
    ax.plot(np.arange(4), feve_lurz_model.mean(0), 'o-', c=c2, label='sensorium model (2022)', markersize=3, alpha=0.6)
    for i in range(nmouse):
        ax.plot(np.arange(4), feve_our_model[i], c='gray', alpha=0.3)
        ax.plot(np.arange(4), feve_lurz_model[i], c=c2, alpha=0.2)
    ax.axhline(feve_LN, color='gray', linestyle='--')
    ax.text(1, 0.2, "our model", transform=ax.transAxes, ha="right", color='black')
    ax.text(1, 0.1, "sensorium model (2022)", transform=ax.transAxes, ha="right", color=c2, alpha=1)
    ax.text(1, 0.42, 'LN baseline', transform=ax.transAxes, ha="right", color='gray', fontsize=12)
    ax.set_xlabel('# of conv layers')
    ax.set_ylabel('FEVE')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.1, 3.1)
    ax.set_xticks([0, 1, 2, 3], [1, 2, 3, 4])
    ax.set_yticks(np.arange(0, 0.81, 0.2), labels=[f'{i:.2f}' for i in np.arange(0, 0.81, 0.2)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(0.8*np.diff(ax.get_xlim())/np.diff(ax.get_ylim()), adjustable='box')
    return il

def panelg_readouts(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, :2])
    ax1.set_title('Readout pooling weights')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax = fig.add_axes([pos[0]+0.04, pos[1]-0.0, pos[2]-0.03, pos[3]-0.0])
    varexp_ratio = dat['feve_example']
    Wxy = dat['Wxy_example']
    fev = dat['fev_example']
    Wxy = dat['Wxy_example']
    Wx = dat['Wx_example']
    Wy = dat['Wy_example']
    nshow = 4
    np.random.seed(9)
    igood = np.where((varexp_ratio > 0.7)&(fev > 0.15))[0]
    idxes = np.random.choice(igood, nshow, replace=False)

    inner = gridspec.GridSpecFromSubplotSpec(3,4,
                    subplot_spec=grid[2,:2], wspace=0.1, hspace=0.2)

    for i in range(nshow):
        ax = fig.add_subplot(inner[2, i])
        ax.imshow(Wxy[idxes[i]], cmap='gray')
        if i == 0:
            ax.text(-28, len(Wxy[0])/2, 'Wxy', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.025-0.005*i, sub_pos.y0-0.02, sub_pos.width-0.01, sub_pos.height])
        ax = fig.add_subplot(inner[0, i])
        ax.plot(Wx[idxes[i], 0, :], color='black')
        ax.set_aspect(0.7/ax.get_data_ratio(), adjustable='box')
        if i == 0:
            ax.text(-len(Wx[0,0])/2, 0.1, 'Wx', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.02-0.005*i, sub_pos.y0-0.02, sub_pos.width-0.0, sub_pos.height])
        ax = fig.add_subplot(inner[1, i])
        ratio = len(Wy[0,0]) / len(Wx[0, 0])
        ax.plot(Wy[idxes[i], 0, :], color='black')
        ax.set_xlim([0-len(Wy[0,0])/2, len(Wx[0,0])-len(Wy[0,0])/2])
        ax.set_aspect(0.7/ax.get_data_ratio(), adjustable='box')
        if i == 0:
            ax.text(-len(Wy[0,0])*1.31, 0.1, 'Wy', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.02-0.005*i, sub_pos.y0-0.02, sub_pos.width-0.0, sub_pos.height])
    return il

def panelh_rfsize_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 2])
    ax1.set_title('Pooling size distribution')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.015, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.025, pos[1]-0.0, pos[2]-0.02, pos[3]-0.0])
    rfsize_all = dat['rfsize_all']
    rfsize_all = np.sqrt(rfsize_all* (270/264)*(65/66)) * 2 
    idx = dat['fev_all'] > 0.15
    ax.hist(rfsize_all[idx], bins=20, color='dimgray', alpha=1, edgecolor='white', linewidth=1)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 5000)
    ax.set_xlabel('Pooling diameter (deg)')
    ax.set_ylabel('# of neurons')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def paneli_monkey_experiment(fig, grid, dat, il, transl):
    ax = plt.subplot(grid[0, 3])
    ax.axis('off')
    il = plot_label(ltr, il, ax, transl, fs_title)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax.set_title('Experiment design')
    return il

def panelj_monkey_activity(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 4])
    ax1.set_title('Example neural activity\n(test images)', y=0.91)
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.01, pos[2]+0.02, pos[3]-0.0])
    fev = dat['monkey_fev']
    feve = dat['monkey_feve']
    spks_test_mean = dat['monkey_spks_test_mean']
    spks_pred_test = dat['monkey_spks_pred_test']

    nn_show = 5
    ineurons_show = np.where((fev>0.3)&(feve>0.7))[0]
    seed = 9
    np.random.seed(seed)
    mycmap = plt.cm.get_cmap('tab20c')
    ineurons_show1 = np.random.choice(ineurons_show, 4, replace=False)
    ineurons_show = np.where((fev>0.5)&(feve>0.5)&(feve<0.6))[0]
    ineurons_show3 = np.random.choice(ineurons_show, 1, replace=False)
    ineurons_show = np.concatenate([ineurons_show1,  ineurons_show3])
    isort = np.argsort(feve[ineurons_show])
    # pos = ax.get_position().bounds
    # ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]+0, pos[3]+0])
    for i in range(nn_show):
        ineuron = ineurons_show[isort[i]]
        neuron_feve = feve[ineuron]
        tmp_neural = spks_test_mean[50:200, ineuron]
        tmp_neural = (tmp_neural - tmp_neural.min()) / (tmp_neural.max() - tmp_neural.min())
        tmp_model = spks_pred_test[50:200, ineuron]
        tmp_model = (tmp_model - tmp_model.min()) / (tmp_model.max() - tmp_model.min())
        tmp_neural /= 1.5
        tmp_model /= 1.5
        if i==0:
            ax.plot(tmp_neural + i+0.1*i, label='true', c='black', linewidth=1)
            ax.plot(tmp_model + i+0.1*i, label='predicted', c=mycmap(0), linewidth=1)
        else:
            ax.plot(tmp_neural + i+0.1*i, c='black', label='_nolegend_', linewidth=1)
            ax.plot(tmp_model + i+0.1*i, c=mycmap(0), label='_nolegend_', linewidth=1)
        ax.text(0, i+0.1*i+0.85, f'Neuron {5-i}, FEVE={neuron_feve:.2f}', fontsize=8, va='center')
    # set all axis off
    ax.axis('off')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, nn_show+1)
    # set xticks off
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(1.025, 1.06))
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    return il

def panelk_monkey_feve_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1,3])
    ax1.set_title('FEVE distribution')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.01, pos[2]-0.02, pos[3]-0.0])
    feve = dat['monkey_feve']
    p = sns.palettes.color_palette('Greys')
    ax.hist(feve, bins=10, color='dimgray', alpha=1, edgecolor='white', linewidth=2)
    ax.scatter(np.mean(feve), 45, s=100, c='dimgray', marker='v', label='mean')
    ax.text(np.mean(feve), 48, f'{np.mean(feve):.2f}', ha='center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 50)
    ax.set_xlabel('FEVE')
    ax.set_ylabel('# of neurons')
    ax.set_xticks(np.arange(0, 1.01, 0.25), [f'{i:.2f}' for i in np.arange(0, 1.01, 0.25)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panell_monkey_depth(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 4])
    ax1.set_title('FEVE change with model depth')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.025, pos[1]+0.01, pos[2]+0.02, pos[3]-0.0])
    vgg_eve = dat['vgg_eve']
    monkey_eve = dat['monkey_depth_eve']
    monkey_LN_feve = dat['monkey_LNmodel_feve']
    import seaborn as sns
    ax.plot(np.arange(4), monkey_eve.mean(0), 'o-', c='black', label='Fullmodel', markersize=3)
    from scipy.stats import sem
    ax.errorbar(np.arange(4), monkey_eve.mean(0), yerr=sem(monkey_eve, 0), color='black', capsize=5)
    ax.plot(4, vgg_eve.mean(), 'o-', c='#4cb938', label='VGG', markersize=3, alpha=0.6)
    ax.errorbar(4, vgg_eve.mean(), yerr=sem(vgg_eve), color='#4cb938', capsize=5, alpha=0.6)
    ax.axhline(monkey_LN_feve, color='gray', linestyle='--')
    ax.text(1, 0.3, "LN baseline", transform=ax.transAxes, ha="right", color='gray')
    ax.set_xlabel('# of conv layers')
    ax.set_ylabel('FEVE')
    ax.set_ylim(0., 0.6)
    ax.set_xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 'VGG'])
    ax.set_yticks(np.arange(0, 0.61, 0.2), [f'{i:.2f}' for i in np.arange(0, 0.61, 0.2)])
    # set right and top axis off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-0.1, 4.2)
    ax.set_aspect(0.75/ax.get_data_ratio(), adjustable='box')
    return il

def panelm_readouts(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 3])
    ax1.set_title('Readout pooling weights')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    eve = dat['monkey_feve']
    fev = dat['monkey_fev']
    Wxy = dat['monkey_Wxy']
    Wx = dat['monkey_Wx']
    Wy = dat['monkey_Wy']
    nshow = 3
    np.random.seed(52)
    igood = np.where((eve > 0.7)&(fev > 0.15))[0]
    idxes = np.random.choice(igood, nshow, replace=False)

    inner = gridspec.GridSpecFromSubplotSpec(3,3,
                    subplot_spec=grid[2,3], wspace=0.1, hspace=0.1, height_ratios=[3,3,4])

    for i in range(nshow):
        # ax[2, i].imshow(Wxy[idxes[i]], cmap='coolwarm_r', vmin=-0.03, vmax=0.03)
        ax = fig.add_subplot(inner[2, i])
        ax.imshow(Wxy[idxes[i]], cmap='gray')
        if i == 0:
            ax.text(-25, len(Wxy[0])/2, 'Wxy', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.03+0.005*i, sub_pos.y0-0.02, sub_pos.width*1.2, sub_pos.height])
        ax = fig.add_subplot(inner[0, i])
        ax.plot(Wx[idxes[i], 0, :], color='black')
        ax.set_aspect(0.7/ax.get_data_ratio(), adjustable='box')
        if i == 0:
            ax.text(-len(Wx[0,0])/2-5, 0.05, 'Wx', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.03+0.005*i, sub_pos.y0-0.02, sub_pos.width*1.2, sub_pos.height])
        ax = fig.add_subplot(inner[1, i])
        ax.plot(Wy[idxes[i], 0, :], color='black')
        ax.set_aspect(0.7/ax.get_data_ratio(), adjustable='box')
        if i == 0:
            ax.text(-len(Wy[0,0])/2-5, 0.05, 'Wy', va='center', ha='left')
        ax.axis('off')
        sub_pos = ax.get_position()
        ax.set_position([sub_pos.x0+0.03+0.005*i, sub_pos.y0-0.02, sub_pos.width*1.2, sub_pos.height])
    return il

def paneln_monkey_rfsize_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 4])
    ax1.set_title('Pooling size distribution')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.04, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.06, pos[1]-0.0, pos[2]-0.02, pos[3]-0.0])
    rfsize_all = dat['monkey_rfsize']
    rfsize_all = np.sqrt(rfsize_all) * 2 * (1.1/80)
    ax.hist(rfsize_all, bins=10, color='dimgray', alpha=1, edgecolor='white', linewidth=1)
    ax.set_ylim(0, 80)
    ax.set_xlabel('Pooling diameter (deg)')
    ax.set_ylabel('# of neurons')
    ax.set_xlim(0, 0.8)
    ax.set_xticks([0, 0.4, 0.8], [f'{i:.1f}' for i in [0, 0.4, 0.8]])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def figure1(dat, root):
    fig = plt.figure(figsize=(15,9))
    grid = plt.GridSpec(3,5, figure=fig, left=0.02, right=0.95, top=0.96, bottom=0.05, 
                        wspace = 0.35, hspace = 0.25)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_experiment(fig, grid, dat, il, transl)
    il = panelb_model(fig, grid, dat, il, transl)
    il = panelc_activity(fig, grid, dat, il, transl)
    # transl = mtransforms.ScaledTranslation(-52 / 72, 7 / 72, fig.dpi_scale_trans)
    il = paneld_feve_distribution(fig, grid, dat, il, transl)
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = panele_5k_30k(fig, grid, dat, il, transl)
    transl = mtransforms.ScaledTranslation(-15 / 72, 7 / 72, fig.dpi_scale_trans)
    il = panelf_depth(fig, grid, dat, il, transl)
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = panelg_readouts(fig, grid, dat, il, transl)
    il = panelh_rfsize_distribution(fig, grid, dat, il, transl)
    il = paneli_monkey_experiment(fig, grid, dat, il, transl)
    il = panelj_monkey_activity(fig, grid, dat, il, transl)
    il = panelk_monkey_feve_distribution(fig, grid, dat, il, transl)
    il = panell_monkey_depth(fig, grid, dat, il, transl)
    transl = mtransforms.ScaledTranslation(-22 / 72, 7 / 72, fig.dpi_scale_trans)
    il = panelm_readouts(fig, grid, dat, il, transl)
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = paneln_monkey_rfsize_distribution(fig, grid, dat, il, transl)
    fig.savefig(os.path.join(root, 'figure1.pdf'), dpi=200) 