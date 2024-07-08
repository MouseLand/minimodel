import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os
import seaborn as sns
from matplotlib import gridspec

def panela_model(fig, grid, dat, il, transl):
    ax = plt.subplot(grid[0, 0])
    ax.axis('off')
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title('Model architecture')
    return il


def panelb_width(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 1])
    ax1.set_title('FEVE change with model width')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.01, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.02+0.02, pos[1]+0.005, pos[2]-0.01, pos[3]-0.0])

    nconv1_list = dat['nconv1']
    nconv2_list = dat['nconv2']
    feve_our_model_vary_width = dat['feve_our_model_vary_width']
    istart = 0
    iend = 10
    mycmap = sns.color_palette("mako", as_cmap=True)
    im = ax.imshow(feve_our_model_vary_width.mean(0)[istart:][:, :iend], cmap=mycmap, vmin=.36, vmax=.74, aspect=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.0235, pad=0.04, label='FEVE', ticks = [0.36, 0.55, 0.74])
    cbar.outline.set_visible(False)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in [0.4, 0.5, 0.6]])
    ax.set_xticks(np.arange(len(nconv2_list)), nconv2_list, rotation=90)
    ax.set_yticks(np.arange(len(nconv1_list[istart:])), nconv1_list[istart:])
    ax.set_ylabel('# of conv1')
    ax.set_xlabel('# of conv2')
    ax.yaxis.set_label_coords(-0.22,0.45)
    ax.scatter(7, 1, marker='x', s=50, c='k', linewidths=2)
    ax.set_title('mouse', loc='center', y=1.)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in [0.36, 0.55, 0.74]], fontsize=10)
    return il

def panelc_width_curve_conv1(fig, grid, dat, il, transl):
    nconv1_list = dat['nconv1']
    nconv2_list = dat['nconv2']
    feve_our_model_vary_width = dat['feve_our_model_vary_width']
    ax1 = plt.subplot(grid[0, 2])
    ax1.set_title('FEVE change with model width (separated)')
    ax1.axis('off')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.04+0.027, pos[1]-0.005, pos[2]-0.01, pos[3]-0.0])

    cmap = sns.color_palette("Blues", as_cmap=True)
    ax.text(390, 0.30, '# conv2', fontsize=12, ha='center')
    for i, nconv2 in enumerate(nconv2_list):
        ax.plot(nconv2_list, feve_our_model_vary_width.mean(0)[:, i], label=f'nconv2={nconv2}',c=cmap((i+3)/(len(nconv2_list)+3)))
        ax.text(360 + 65*(i//5), 0.01*(24-(i%5)*5.5), f'{nconv2}', color=cmap((i+3)/(len(nconv2_list)+3)), ha='center')
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv1')
    ax.set_ylabel('FEVE')
    ax.set_yticks([0, 0.25, 0.50, 0.75])
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def paneld_width_curve_conv2(fig, grid, dat, il, transl):
    nconv1_list = dat['nconv1']
    nconv2_list = dat['nconv2']
    feve_our_model_vary_width = dat['feve_our_model_vary_width']
    ax1 = plt.subplot(grid[0, 3])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.0+0.03, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.0+0.03, pos[1]-0.005, pos[2]-0.01, pos[3]-0.0])

    cmap = sns.color_palette("Purples", as_cmap=True)
    ax.text(390, 0.30, '# conv1', fontsize=12, ha='center')
    for i, nconv1 in enumerate(nconv1_list):
        ax.plot(nconv1_list, feve_our_model_vary_width.mean(0)[i], label=f'nconv1={nconv1}',c=cmap((i+3)/(len(nconv1_list)+3)))
        ax.text(360 + 65*(i//5), 0.01*(24-(i%5)*5.5), f'{nconv1}', color=cmap((i+3)/(len(nconv1_list)+3)), ha='center')
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 450)
    ax.set_yticks([0, 0.25, 0.50, 0.75])
    ax.set_xlabel('# of conv2')
    # remove the y axis as well as the y axis line
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('mouse', x=-0.25, y=0.99)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il


def panele_kernels(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[0, 4])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('Conv1 weights')
    conv1_W = dat['conv1_W']
    inner = gridspec.GridSpecFromSubplotSpec(4,4,
                        subplot_spec=grid[0, 4], wspace=0.2, hspace=0.0)
    # ax1.remove()
    isort = [13,5,6,15,10,12,2,12,1,3,9,14,7,0,11,4]
    for i in range(16):
        ax = fig.add_subplot(inner[i//4, i%4])
        im = ax.imshow(conv1_W[isort[i]], cmap='RdBu', vmin=-0.2, vmax=0.2)
        ax.axis('off')
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        yadd = (i//4) * 0.005
        new_pos = [pos.x0-0.01, pos.y0 - 0.04 + yadd, pos.width-0.00, pos.height]  
        ax.set_position(new_pos)
    pos = ax.get_position()
    ax = fig.add_subplot(grid[0, 4])    
    ax.set_title('mouse', x=0.22, y=0.86)
    ax.axis('off')
    return il

def panelf_monkey_model_width(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 1])
    # ax1.set_title('EVE change with model width')
    # il = plot_label(ltr, il, ax1, transl, fs_title)

    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.01, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')

    ax = fig.add_axes([pos[0]+0.02+0.02, pos[1]+0.035, pos[2]-0.01, pos[3]-0.0])
    eve_all_models = dat['monkey_all_width_eve']
    nconv1_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
    nconv2_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]

    mycmap = sns.color_palette("mako", as_cmap=True)
    istart = 0
    im = ax.imshow(eve_all_models, cmap=mycmap, vmin=0.43, vmax=0.57, aspect=1)
    cbar = plt.colorbar(im, fraction=0.0235, pad=0.04, label='FEVE', ticks = [0.43, 0.5, 0.57])
    cbar.outline.set_visible(False)
    # cbar.ax.tick_params(labelsize='small')
    ax.set_xticks(np.arange(len(nconv2_list)), nconv2_list, rotation=90)
    ax.set_yticks(np.arange(len(nconv1_list[istart:])), nconv1_list[istart:])
    ax.set_ylabel('# of conv1')
    ax.set_xlabel('# of conv2')
    ax.yaxis.set_label_coords(-0.22,0.45)
    ax.scatter(7, 1, marker='x', s=50, c='k', linewidths=2)
    ax.set_title('monkey', loc='center', y=1.)
    cbar.ax.set_yticklabels([f'{i:.2f}' for i in [0.43, 0.5, 0.57]], fontsize=10)
    return il

def panelg_monkey_width_curve_conv1(fig, grid, dat, il, transl):
    nconv1_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
    nconv2_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
    feve_our_model_vary_width = dat['monkey_all_width_eve']
    ax1 = plt.subplot(grid[1, 2])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.02+0.02, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.04+0.027, pos[1]+0.025, pos[2]-0.01, pos[3]-0.0])

    cmap = sns.color_palette("Blues", as_cmap=True)
    for i, nconv2 in enumerate(nconv2_list):
        ax.plot(nconv2_list, feve_our_model_vary_width[:, i], label=f'nconv2={nconv2}',c=cmap((i+3)/(len(nconv2_list)+3)))
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv1')
    ax.set_ylabel('FEVE')
    ax.set_yticks([0, 0.25, 0.50, 0.75])
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def panelh_monkey_width_curve_conv2(fig, grid, dat, il, transl):
    nconv1_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
    nconv2_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
    feve_our_model_vary_width = dat['monkey_all_width_eve']
    ax1 = plt.subplot(grid[1, 3])
    # ax1.set_title('EVE change with model width')
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.0+0.03, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.0+0.03, pos[1]+0.025, pos[2]-0.01, pos[3]-0.0])

    cmap = sns.color_palette("Purples", as_cmap=True)
    for i, nconv1 in enumerate(nconv1_list):
        ax.plot(nconv1_list, feve_our_model_vary_width[i]*100, label=f'nconv1={nconv1}',c=cmap((i+3)/(len(nconv2_list)+3)))
    ax.set_ylim(0, 75)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv2')
    ax.set_yticks([0, 25, 50, 75])
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('monkey', x=-0.23, y=0.99)
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')
    return il

def paneli_monkey_kernels(fig, grid, dat, il, transl):
    ax1 = plt.subplot(grid[1, 4])
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    conv1_W = dat['monkey_conv1_W']
    isort = [0,1,11,9,5,8,13,7,4,6,12,14,15,10,2,3]
    inner = gridspec.GridSpecFromSubplotSpec(4,4,
                        subplot_spec=grid[1,4], wspace=0.2, hspace=0.0)

    for i in range(16):
        ax = fig.add_subplot(inner[i//4, i%4])
        im = ax.imshow(conv1_W[isort[i]], cmap='RdBu', vmin=-0.2, vmax=0.2)
        ax.axis('off')
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        yadd = (i//4) * 0.005
        new_pos = [pos.x0-0.01, pos.y0 - 0.01 + yadd, pos.width-0.00, pos.height]  # Adjust the 0.05 as needed
        ax.set_position(new_pos)
    pos = ax.get_position()
    ax = fig.add_subplot(grid[1, 4])    
    ax.set_title('monkey', x=0.22, y=1.0)
    ax.axis('off')
    return il

def panelj_model(fig, grid, dat, il, transl):
    ax = plt.subplot(grid[2, 0])
    ax.axis('off')
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title('Classification tasks')
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.0, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    return il

def panel_texturenet_acc(fig, grid, dat, il, transl, nchan):
    ax1 = plt.subplot(grid[2, 1])
    ax1.set_title('Texture classification accuracy')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.07, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.09, pos[1]+0.02, pos[2]-0.01, pos[3]-0.0])
    texturenet_acc = dat['texturenet_accuracy']
    mycmap = sns.color_palette("magma", as_cmap=True)
    im = ax.imshow(texturenet_acc.T, vmin=0, vmax=80, cmap=mycmap)
    ax.set_xlabel("# of conv2")
    ax.set_ylabel("# of conv1")
    ax.set_xticks(np.arange(0, len(nchan)))
    ax.set_yticks(np.arange(0, len(nchan)))
    ax.set_xticklabels(nchan, rotation=90)
    ax.set_yticklabels(nchan)
    cbar = plt.colorbar(im, label="accuracy (%)", ticks=np.arange(0, 81, 20), ax=ax, fraction=0.0235, pad=0.04)
    cbar.outline.set_visible(False)
    return il

def panelk_texturenet_acc(fig, grid, dat, il, transl, nchan):
    nconv1_list = dat['nconv1']
    nconv2_list = dat['nconv2']
    texturenet_acc = dat['texturenet_accuracy']
    inner = gridspec.GridSpecFromSubplotSpec(2,1,
                        subplot_spec=grid[2,2], wspace=0., hspace=0.1)
    ax = fig.add_subplot(inner[0, 0])
    cmap = sns.color_palette("Blues", as_cmap=True)
    xloc = 500
    for i, nconv1 in enumerate(nconv1_list):
        ax.plot(nconv1_list, texturenet_acc[i, :], label=f'nconv1={nconv1}',c=cmap((i+3)/(len(nconv1_list)+3)))
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv1')
    ax.set_ylabel('accuracy (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 200, 400])
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    pos = ax.get_position()
    new_pos = [pos.x0+0.09, pos.y0+0.02, pos.width-0.04, pos.height]  # Adjust the 0.05 as needed
    ax.set_position(new_pos)
    ax = fig.add_subplot(inner[1, 0])
    cmap = sns.color_palette("Purples", as_cmap=True)
    xloc = 500
    for i, nconv2 in enumerate(nconv2_list):
        ax.plot(nconv1_list, texturenet_acc[:, i], label=f'nconv2={nconv2}',c=cmap((i+3)/(len(nconv2_list)+3)))
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv2')
    ax.set_ylabel('accuracy (%)')
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 200, 400])
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    pos = ax.get_position()
    new_pos = [pos.x0+0.09, pos.y0+0.02, pos.width-0.04, pos.height]  
    ax.set_position(new_pos)
    return il

def panel_imagenet_acc(fig, grid, dat, il, transl, nchan):
    ax1 = plt.subplot(grid[2, 3])
    ax1.set_title('ImageNet classification accuracy')
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.03, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.05, pos[1]+0.02, pos[2]-0.01, pos[3]-0.0])
    imagenet_accuracy = dat['imagenet_accuracy']

    mycmap = sns.color_palette("magma", as_cmap=True)
    im = ax.imshow(imagenet_accuracy.T, vmin=0, vmax=40, cmap=mycmap)
    ax.set_xlabel("# of conv2")
    ax.set_ylabel("# of conv1")
    ax.set_xticks(np.arange(0, len(nchan)))
    ax.set_yticks(np.arange(0, len(nchan)))
    ax.set_xticklabels(nchan, rotation=90)
    ax.set_yticklabels(nchan)
    cbar = plt.colorbar(im, label="accuracy (%)", ticks=np.arange(0, 45, 10), ax=ax, fraction=0.0235, pad=0.04)
    cbar.outline.set_visible(False)
    return il

def panell_imagenet_acc(fig, grid, dat, il, transl, nchan):
    nconv1_list = dat['nconv1']
    nconv2_list = dat['nconv2']
    ax1 = plt.subplot(grid[2, 4])
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.03, pos[1]+0.01, pos[2]-0.0, pos[3]-0.0])
    ax1.axis('off')
    imagenet_accuracy = dat['imagenet_accuracy']
    inner = gridspec.GridSpecFromSubplotSpec(2,1,
                        subplot_spec=grid[2,4], wspace=0., hspace=0.1)
    ax = fig.add_subplot(inner[0, 0])
    cmap = sns.color_palette("Blues", as_cmap=True)
    xloc = 500
    # ax.text(xloc+35, 47, '# conv2', fontsize=12, ha='center')
    for i, nconv1 in enumerate(nconv1_list):
        ax.plot(nconv1_list, imagenet_accuracy[i, :], label=f'nconv1={nconv1}',c=cmap((i+3)/(len(nconv1_list)+3)))
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv1')
    ax.set_ylabel('accuracy (%)')
    ax.set_yticks([0, 25, 50])
    ax.set_xticks([0, 200, 400])
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    pos = ax.get_position()
    new_pos = [pos.x0+0.05, pos.y0+0.02, pos.width-0.04, pos.height]  # Adjust the 0.05 as needed
    ax.set_position(new_pos)
    ax = fig.add_subplot(inner[1, 0])
    cmap = sns.color_palette("Purples", as_cmap=True)
    xloc = 500
    # ax.text(xloc+35, 47, '# conv1', fontsize=12, ha='center')
    for i, nconv2 in enumerate(nconv2_list):
        ax.plot(nconv1_list, imagenet_accuracy[:, i], label=f'nconv2={nconv2}',c=cmap((i+3)/(len(nconv2_list)+3)))
    ax.set_ylim(0, 50)
    ax.set_xlim(0, 450)
    ax.set_xlabel('# of conv2')
    ax.set_ylabel('accuracy (%)')
    ax.set_yticks([0, 25, 50])
    ax.set_xticks([0, 200, 400])
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')
    pos = ax.get_position()
    new_pos = [pos.x0+0.05, pos.y0+0.02, pos.width-0.04, pos.height]  # Adjust the 0.05 as needed
    ax.set_position(new_pos)
    return il


def figure2(dat, root):
    fig = plt.figure(figsize=(14,9))
    grid = plt.GridSpec(3,5, figure=fig, left=0.02, right=0.97, top=0.96, bottom=0.02, 
                        wspace = 0.4, hspace = 0.25, height_ratios=[4,4,4])   
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_model(fig, grid, dat, il, transl)
    il = panelb_width(fig, grid, dat, il, transl)
    il = panelc_width_curve_conv1(fig, grid, dat, il, transl)
    il = paneld_width_curve_conv2(fig, grid, dat, il, transl)
    il = panele_kernels(fig, grid, dat, il, transl)
    il = panelf_monkey_model_width(fig, grid, dat, il, transl)
    il = panelg_monkey_width_curve_conv1(fig, grid, dat, il, transl)
    il = panelh_monkey_width_curve_conv2(fig, grid, dat, il, transl)
    il = paneli_monkey_kernels(fig, grid, dat, il, transl)
    il = panelj_model(fig, grid, dat, il, transl)
    nchan = [8, 16, 32, 64, 128, 192, 256, 320, 384, 448]
    il = panel_texturenet_acc(fig, grid, dat, il, transl, nchan)
    il = panelk_texturenet_acc(fig, grid, dat, il, transl, nchan)
    il = panel_imagenet_acc(fig, grid, dat, il, transl, nchan)
    il = panell_imagenet_acc(fig, grid, dat, il, transl, nchan)
    fig.savefig(os.path.join(root, 'figure2.pdf'), dpi=200) 