import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os
import seaborn as sns
from matplotlib import GridSpec

def panela_vary_images_mouse(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Performance change with number of train images')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.01, pos[2]-0.015, pos[3]+0])
    all_feve = dat['feve_all_nstim']
    nstim_list = dat['nstim']
    nmouse = 6
    for imouse in range(nmouse):
        ax.plot(nstim_list, all_feve[imouse], lw=2, color='gray', alpha=0.5)
    ax.plot(nstim_list, all_feve.mean(axis=0), 'k', label='all', lw=1)
    # add sem error bar
    from scipy.stats import sem
    ax.errorbar(nstim_list, all_feve.mean(axis=0), yerr=sem(all_feve, axis=0), color='k', capsize=2, linewidth=1)
    ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('# of train images')
    ax.set_ylabel('FEVE')
    ax.set_yticks(np.arange(0, 1.1, 0.25), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.25)])
    ax.set_ylim(-0.05, 1.00)
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    xticks = [1e3, 1e4, 3e4] 
    xtick_labels = ['10$^3$', '10$^4$', '3x10$^4$']  
    plt.xticks(xticks, xtick_labels, fontname='Arial')
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    ax.set_title('mouse', loc='center', y=0.95)
    return il

def panela_vary_images_monkey(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 3])
    ax1.axis('off')
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('Performance as a function of images')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.01, pos[2]-0.015, pos[3]+0])
    all_feve = dat['monkey_feve_all_nstim']
    nstim_list = dat['monkey_nstim']
    ax.plot(nstim_list, all_feve.mean(axis=1), 'k', label='all', lw=1)
    # add sem error bar
    from scipy.stats import sem
    ax.errorbar(nstim_list, all_feve.mean(axis=1), yerr=sem(all_feve, axis=1), color='k', capsize=2, linewidth=1)
    ax.set_xscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('# of train images')
    ax.set_ylabel('FEVE')
    ax.set_yticks(np.arange(0, 1.1, 0.25), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.25)])
    ax.set_ylim(-0.05, 1.00)
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    xticks = [5e2, 1e3, 5e3]  
    xtick_labels = ['5x10$^2$', '10$^3$', '5x10$^3$'] 
    plt.xticks(xticks, xtick_labels)
    # set equal aspect ratio considering the log
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    ax.set_title('monkey', loc='center', y=0.95)
    return il

def panelb_vary_neurons_mouse(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 0])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Performance change with number of neurons')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.01, pos[2]-0.015, pos[3]+0])
    NNs = dat['NNs']
    all_feve = dat['feve_all_nn']
    nmouse = 6
    for imouse in range(nmouse):
        ax.plot(NNs, all_feve[imouse], lw=2, color='gray', alpha=0.5)
    ax.plot(NNs, all_feve.mean(axis=0), 'k', lw=1)
    from scipy.stats import sem
    ax.errorbar(NNs, all_feve.mean(axis=0), yerr=sem(all_feve, axis=0), color='k', capsize=2, lw=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-0.05, 1)
    # set yticks
    ax.set_yticks(np.arange(0, 1.1, 0.25), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.25)])
    ax.set_xscale('log')
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('FEVE')
    ax.set_aspect(0.8 / ax.get_data_ratio())
    xticks = [1e0, 1e1, 1e2, 1e3] 
    xtick_labels = ['10$^0$', '10$^1$', '10$^2$', '10$^3$'] 
    plt.xticks(xticks, xtick_labels, fontname='Arial')
    ax.set_title('mouse', loc='center', y=0.95)
    return il

def panelb_vary_neurons_monkey(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 1])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.01, pos[2]-0.015, pos[3]+0])
    nmouse = 2
    monkey_id_values = [4, 34]
    NNs = dat['monkey_NNs']
    feve_all = dat['monkey_feve_all_nn']
    # set font size
    plt.rc('font', size=12) 
    feve_mean = []
    feve_sem = []
    from scipy.stats import sem
    for i in range(len(NNs)):
        feve_mean.append(np.mean(feve_all[i]))
        feve_sem.append(sem(feve_all[i]))
    ax.plot(NNs, feve_mean, 'k', lw=1)
    ax.errorbar(NNs, feve_mean, yerr=feve_sem, color='k', capsize=2, linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-0.05, 1)
    # set yticks
    ax.set_yticks(np.arange(0, 1.1, 0.25), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.25)])
    ax.set_xscale('log')
    ax.set_xlabel('# of neurons')
    ax.set_ylabel('FEVE')
    ax.set_aspect(0.8 / ax.get_data_ratio())
    xticks = [1e0, 1e1, 1e2]  
    xtick_labels = ['10$^0$', '10$^1$', '10$^2$'] 
    plt.xticks(xticks, xtick_labels, fontname='Arial')
    ax.set_title('monkey', loc='center', y=0.95)
    return il

def panelc_model(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 0])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Minimodel architecture')
    return il

def paneld_minimode_fullmodel(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 1])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Minimodel vs 16-320 model')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.01, pos[1]-0.02, pos[2]-0.0, pos[3]+0])
    nmouse = 6
    fev_all = dat['fev_all']
    fullmodel_feve_all = dat['fullmodel_feve_all']
    minimodel_feve_all = dat['minimodel_feve_all']
    mycmap = sns.color_palette("rocket", as_cmap=True)
    x = []
    y = []
    c = []
    for i in range(nmouse):
        idx = np.where(fev_all[i] > 0.15)[0]
        x.append(fullmodel_feve_all[i][idx])
        y.append(minimodel_feve_all[i])
        c.append(fev_all[i][idx])
    x = np.concatenate(x)   
    y = np.concatenate(y)
    c = np.concatenate(c)
    sc = ax.scatter(x, y, alpha=0.1, c=c, cmap=mycmap, s=1, vmin=0.15, vmax=0.35, rasterized=True)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.plot([0, 1], [0, 1], '--', c='black')
    ax.set_xlabel('16-320 model FEVE')
    ax.set_ylabel('Minimodel FEVE')
    # add colorbar and set the size
    cbar = plt.colorbar(sc, ax=ax, ticks=[0.15,0.35], fraction=0.02, pad=0.01)
    cbar.set_label('FEV', rotation=270, labelpad=15, fontsize=10)
    cbar.solids.set(alpha=1)
    cbar.outline.set_visible(False)
    # set the fontsize of the ticks
    cbar.ax.tick_params(labelsize=10)
    ax.set_yticks(np.arange(0, 1.1, 0.5), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.5)])
    ax.set_xticks(np.arange(0, 1.1, 0.5), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.5)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('mouse', loc='center', y=0.95)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def pannele_mean_minimodel_fullmodel(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 2])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Average performance')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.03, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.04, pos[1]-0.02, pos[2]-0.0, pos[3]+0])
    fev_all = dat['fev_all']
    nmouse = 6
    minimodel_feve_avg = np.zeros(nmouse)
    fullmodel_feve_avg = np.zeros(nmouse)
    fullmodel_feve_all = dat['fullmodel_feve_all']
    minimodel_feve_all = dat['minimodel_feve_all']
    thresh = 0.15
    for i in range(nmouse):
        idx = np.where(fev_all[i] > thresh)[0]
        minimodel_feve_avg[i] = minimodel_feve_all[i].mean()
        fullmodel_feve_avg[i] = fullmodel_feve_all[i][idx].mean()
    plot_model_names = ['16-320\nmodel', 'minimodel']
    plot_model_mean = [fullmodel_feve_avg.mean(), minimodel_feve_avg.mean()]
    plot_model_std = [fullmodel_feve_avg.std(), minimodel_feve_avg.std()]
    for i in range(6):
            label = '16-320 model' if i==0 else 'minimodel'
            ax.plot(np.arange(2), [fullmodel_feve_avg[i], minimodel_feve_avg[i]], '-o', alpha=0.5, label='mouse'+str(i+1), color='gray')
    ax.errorbar(np.arange(2), plot_model_mean, yerr=plot_model_std, fmt='-', color='k', capsize=2, lw=1)
    ax.set_ylabel('Variance explained (FEVE)')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(2), plot_model_names)
    ax.set_yticks(np.arange(0, 1.1, 0.25), labels=[f'{i:.2f}' for i in np.arange(0, 1.1, 0.25)])
    ax.set_aspect(1.2/ax.get_data_ratio(), adjustable='box')  
    from scipy.stats import ttest_rel
    t, p = ttest_rel(fullmodel_feve_avg, minimodel_feve_avg)
    ax.set_title('mouse', loc='center', y=0.95)
    return il

def panelf_valid_wc_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 3])
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('# of conv2 distribution')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.01, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.0, pos[3]+0])
    minimodel_wc_all = dat['minimodel_wc_all']
    n_valid_wc_all = []
    nmouse = 6
    for i in range(nmouse):
        wc_all = minimodel_wc_all[i]
        n_valid_wc = np.sum(np.abs(wc_all) > 0.01, axis=-1)
        n_valid_wc_all.append(n_valid_wc.squeeze())

    # mycmap = sns.color_palette("crest", as_cmap=True)
    mycmap = sns.color_palette("rocket", as_cmap=True)
    for i in range(nmouse):
        ax.hist(n_valid_wc_all[i], 15, alpha=1.0, label=f'mouse{i+1}', color=mycmap(i/6), histtype='step')
        ax.text(0.7, 0.9-0.08*i, f'mouse{i+1}', transform=ax.transAxes, color=mycmap(i/6))
    ax.set_ylim(0, 1250)
    ax.set_xlim(0, 64)
    # legend outside the plot
    ax.set_xlabel('# of conv2')
    ax.set_ylabel('# of neurons')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('mouse', loc='center', y=0.95)
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panelg_monkey_minimodel_fullmodel(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 1])
    ax1.axis('off')
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('fullmodel vs minimodel')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.01, pos[1]-0.0, pos[2]-0.0, pos[3]+0])

    sigvar_all = dat['monkey_fev_all']
    # fullmodel_feve_all = data_dict['monkey_fullmodel_feve_all']
    fullmodel_feve_all = dat['monkey_feve_all']
    minimodel_feve_all = dat['monkey_minimodel_feve_all']
    mycmap = sns.color_palette("rocket", as_cmap=True)
    sc = ax.scatter(fullmodel_feve_all, minimodel_feve_all, alpha=0.5, c=sigvar_all, cmap=mycmap, s=10, vmin=0.15, vmax=0.35, rasterized=True)
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.2)
    ax.plot([0, 1], [0, 1], '--', c='black')
    ax.set_xlabel('16-320 model FEVE')
    ax.set_ylabel('Minimodel FEVE')
    cbar = plt.colorbar(sc, ax=ax, ticks=[0.15, 0.35], fraction=0.02, pad=0.01)
    cbar.set_label('FEV', rotation=270, labelpad=15, fontsize=10)
    cbar.solids.set(alpha=1)
    cbar.outline.set_visible(False)
    # set the fontsize of the ticks
    cbar.ax.tick_params(labelsize=10)
    ax.set_yticks(np.arange(0, 1.1, 0.5), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.5)])
    ax.set_xticks(np.arange(0, 1.1, 0.5), [f'{x:.2f}' for x in np.arange(0, 1.1, 0.5)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    ax.set_title('monkey', loc='center', y=0.95)
    return il

def panelh_mean_monkey_minimodel_fullmodel(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 2])
    ax1.axis('off')
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('minimodel vs fullmodel')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.03, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.04, pos[1]-0.0, pos[2]-0.0, pos[3]+0])
    fullmodel_feve_all = dat['monkey_feve_all']
    minimodel_feve_all = dat['monkey_minimodel_feve_all']
    plot_model_names = ['16-320\nmodel', 'minimodel']
    plot_model_mean = [fullmodel_feve_all.mean(), minimodel_feve_all.mean()]
    from scipy.stats import sem
    plot_model_sem = [sem(fullmodel_feve_all), sem(minimodel_feve_all)]
    ax.errorbar(np.arange(2), plot_model_mean, yerr=plot_model_sem, fmt='-', color='k', capsize=2, lw=1)
    ax.set_ylabel('Variance explained (FEVE)')
    ax.set_ylim(0, 0.8)
    ax.set_xlim(-0.5, 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(np.arange(2), plot_model_names)
    ax.set_yticks(np.arange(0, 1.1, 0.25), labels=[f'{i:.2f}' for i in np.arange(0, 1.1, 0.25)])
    ax.set_aspect(1.2/ax.get_data_ratio(), adjustable='box')  
    from scipy.stats import ttest_rel
    t, p = ttest_rel(fullmodel_feve_all, minimodel_feve_all)
    ax.set_title('monkey', loc='center', y=0.95)
    return il

def paneli_monkey_valid_wc_distribution(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[2, 3])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.0, pos[2]-0.0, pos[3]+0])
    Wc_all = dat['monkey_wc_all']
    monkey_id = dat['monkey_id']
    NN = 166
    n_pos_wc = np.zeros(NN)
    n_neg_wc = np.zeros(NN)
    for i in range(166):
        n_pos_wc[i] = np.sum(Wc_all>0.01)
        n_neg_wc[i] = np.sum(Wc_all<-0.01)

    n_valid_Wc = np.sum(np.abs(Wc_all)>0.01, axis=1)

    ids = np.unique(monkey_id)
    cmap_vals = [0.2, 0.5]
    mycmap = sns.color_palette("rocket", as_cmap=True)
    for i, id in enumerate(ids):
        ax.hist(n_valid_Wc[monkey_id == id], bins=7, alpha=1, label=f'monkey {i+1}', color=mycmap(cmap_vals[i]), histtype='step')
    # ax.legend(fontsize=14)
    ax.text(42, 25, 'monkey 1', color=mycmap(cmap_vals[0]))
    ax.text(42, 22, 'monkey 2', color=mycmap(cmap_vals[1]))
    ax.set_xlabel('# of conv2')
    ax.set_ylabel('# of neurons')
    ax.set_ylim(0, 50)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 64)
    ax.set_xticks([0, 16, 32, 48, 64])
    ax.set_title('monkey', loc='center', y=0.95)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def figure3(dat, root):
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['mathtext.rm'] = 'Arial'
    fig = plt.figure(figsize=(14,9))
    grid = plt.GridSpec(3,4, figure=fig, left=0.05, right=0.92, top=0.94, bottom=0.08, 
                        wspace = 0.35, hspace = 0.45)  
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    
    il = panelb_vary_neurons_mouse(fig, grid, dat, il, transl)
    il = panelb_vary_neurons_monkey(fig, grid, dat, il, transl)
    il = panela_vary_images_mouse(fig, grid, dat, il, transl)
    il = panela_vary_images_monkey(fig, grid, dat, il, transl)
    il = panelc_model(fig, grid, dat, il, transl)
    il = paneld_minimode_fullmodel(fig, grid, dat, il, transl)
    il = pannele_mean_minimodel_fullmodel(fig, grid, dat, il, transl)
    il = panelf_valid_wc_distribution(fig, grid, dat, il, transl)
    il = panelg_monkey_minimodel_fullmodel(fig, grid, dat, il, transl)
    il = panelh_mean_monkey_minimodel_fullmodel(fig, grid, dat, il, transl)
    il = paneli_monkey_valid_wc_distribution(fig, grid, dat, il, transl)
    fig.savefig(os.path.join(root, 'figure3.pdf'), dpi=200) 