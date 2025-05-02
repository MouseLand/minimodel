import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os


def panela_model(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Minimodel structure')
    ax1.axis('off')
    return il


def panela_channel_activity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('Minimodel structure')
    ax1.axis('off')
    pos = ax1.get_position().bounds
    # ax1.set_position([pos[0]+0.01, pos[1], pos[2], pos[3]])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.0, pos[2]-0.03, pos[3]+0])
    nchan = 22
    # fig, ax = plt.subplots(1, 1, figsize=(3,3))
    wxy_fv = data_dict['wxy_fv']
    model_output = data_dict['model_output']

    cmap = plt.get_cmap('plasma', 9)
    for i in range(8):
        ic = i if i<6 else -(8-i)
        sorted_fv = np.sort(wxy_fv[:, ic])[::-1]
        ic = i+1 if i<6 else nchan+1-(8-i)
        ax.plot(sorted_fv[:200], color=cmap(i), label=f'channel {ic}')
        if i ==7:
            ax.text(0.5, 0.99-5*0.066, f'channel K', color=cmap(i), transform=ax.transAxes)
        elif i<4:
            ax.text(0.5, 0.99-i*0.066, f'channel {ic}', color=cmap(i), transform=ax.transAxes)
        # ax.set_axis_off()
        # plt.savefig(f'./outputs/{model_name}_channel_{ic}_activities.pdf', bbox_inches='tight', dpi=200)
    # set right and top spine invisible
    sorted_output = np.sort(model_output)[::-1]
    ax.plot(sorted_output[:200], color='black', label='model output')
    ax.text(0.5, 0.99-6*0.066, f'model output', color='black', transform=ax.transAxes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Stimuli (sorted by activity)')
    ax.set_ylabel('Activity')
    # set aspect ratio
    # ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # add a bar on the top left corner and add text channel maximum stimuli
    ax.text(0.08, 1.02, 'maximum\nstimuli', transform=ax.transAxes, ha='center')
    ax.set_ylim([1.8, 7])
    ax.plot([1, 16], [7, 7], color='black', linewidth=10)
    # set aspect ratio
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il



def panelb_high_catvar(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Mouse high category variance neuron example')
    ax1.axis('off')
    return il

def panelc_high_catvar_monkey(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Monkey high category variance neuron example')
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.1, pos[1], pos[2], pos[3]])
    return il

def paneld_low_catvar(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Mouse low category variance neuron example')
    ax1.axis('off')
    return il

def panele_low_catvar_monkey(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Monkey low category variance neuron example')
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.1, pos[1], pos[2], pos[3]])
    return il



def figure5(dat, root):
    fig = plt.figure(figsize=(14,14))
    grid = plt.GridSpec(3,3, figure=fig, left=0.02, right=0.95, top=0.96, bottom=0.05, 
                        wspace = 0.35, hspace = 0.25, height_ratios=[1,1,1], width_ratios=[2,1,1])
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_model(fig, grid, dat, il, transl)
    il = panela_channel_activity(fig, grid, dat, il, transl)
    il = panelb_high_catvar(fig, grid, dat, il, transl)
    il = panelc_high_catvar_monkey(fig, grid, dat, il, transl)
    il = paneld_low_catvar(fig, grid, dat, il, transl)
    il = panele_low_catvar_monkey(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'figure5.pdf'), dpi=200) 