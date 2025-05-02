import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from fig_utils import *
import os
from matplotlib import scale as mscale
from fig_utils import AsymScale
mscale.register_scale(AsymScale)
import seaborn as sns
from matplotlib import gridspec
import matplotlib.patches as patches

######################################################################## FIGURE 2 ########################################################################
def get_region(iarea, region):

    if region == 'all':
        ix = np.isin(iarea, np.arange(10)) 
    if region == 'V1':
        ix = np.isin(iarea, [8]) 
    if region == 'medial':
        ix = np.isin(iarea, [0,1,2,9]) 
    if region == 'anterior':
        ix = np.isin(iarea, [3,4]) 
    if region == 'lateral':
        ix = np.isin(iarea, [5,6,7]) 
    return ix

def panela_xretinotopy(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[0,mouse_id])
    if mouse_id == 0:
        il = plot_label(ltr, il, ax1, transl, fs_title)
        ax1.set_title('Horizontal retinotopy (full field of view)')
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]+0.0, pos[3]+0.0])
    iregion = data_dict['iregion']
    iarea = data_dict['iarea'][mouse_id]
    xy = data_dict['xy'][mouse_id]
    out = data_dict['out'][mouse_id]
    jxy = data_dict['jxy'][mouse_id]

    v1 = get_region(iarea, "V1")
    ax.set_title(f"mouse {mouse_id+1}", loc='center')
    # input image resolution: cutting from 22x88 to 22x50 (only keep the left part)
    ax.scatter(xy[v1,1], xy[v1,0], c= jxy[v1,0] * (270/88)-135, s = 0.5, vmax=50* (270/88)-135, vmin=-135, cmap = 'gist_ncar', rasterized=True)
    # ax.scatter(xy[v1,1], xy[v1,0], c= jxy[v1,1], s = 0.5, vmax=25, vmin=0, cmap = 'gist_ncar', rasterized=True)
    if mouse_id == 1:
        xlims = ~(xy[:,1] > -1000) 
        ylims = ~(xy[:,0] < 450)
        selected_neurons = v1 * xlims * ylims
        ax.scatter(xy[selected_neurons,1], xy[selected_neurons,0], c= jxy[selected_neurons,0]* (270/88)-135, s = 0.5, vmax=50* (270/88)-135, vmin=0-135, cmap = 'gist_ncar', rasterized=True)
    if mouse_id == 5:
        lat = np.isin(iarea, [5]) 
        sc = ax.scatter(xy[lat,1], xy[lat,0], c= jxy[lat,0]* (270/88)-135, s = 0.5, vmax=50* (270/88)-135, vmin=0-135, cmap = 'gist_ncar', rasterized=True)
        # ax[1,i].scatter(xy[lat,1], xy[lat,0], c= jxy[lat,1], s = 0.5, vmax=25, vmin=0, cmap = 'gist_ncar', rasterized=True)
        ax.plot(out[8][:,1], out[8][:,0], '-k')
        ax.plot(out[5][:,1], out[5][:,0], '-k')
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.set_label('Azimuth (deg)')
    else:
        ax.plot(out[8][:,1], out[8][:,0], '-k')

    rect0a = patches.Rectangle((-2750,400),742,636,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect0b = patches.Rectangle((-2620,1036),540,636,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect1a = patches.Rectangle((-2100,900),680,420,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect1b = patches.Rectangle((-2400,1320),680,420,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect3 = patches.Rectangle((-2450,500),672,412,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect4 = patches.Rectangle((-2550,700),680,417,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect5 = patches.Rectangle((-2700,800),664,414,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect6 = patches.Rectangle((-1900,850),1426,578,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rects0 = [rect0a,rect0b]
    rects1 = [rect1a,rect1b]
    rects = [rects0, rects1, rect3, rect4, rect5, rect6]
    if  mouse_id <= 1:
        r = rects[mouse_id]
        ax.add_patch(r[0])
        ax.add_patch(r[1])
    elif mouse_id > 1:
        ax.add_patch(rects[mouse_id])
    ax.set_aspect('equal')
    ax.set_axis_off()
    return il

def panelb_yretinotopy(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[1,mouse_id])
    if mouse_id == 0:
        il = plot_label(ltr, il, ax1, transl, fs_title)
        ax1.set_title('Vertical retinotopy (full field of view)')
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]+0.0, pos[3]+0.0])
    iregion = data_dict['iregion']
    iarea = data_dict['iarea'][mouse_id]
    xy = data_dict['xy'][mouse_id]
    out = data_dict['out'][mouse_id]
    jxy = data_dict['jxy'][mouse_id]

    v1 = get_region(iarea, "V1")
    ax.set_title(f"mouse {mouse_id+1}", loc='center')
    # ax.scatter(xy[v1,1], xy[v1,0], c= jxy[v1,0], s = 0.5, vmax=50, vmin=0, cmap = 'gist_ncar', rasterized=True)
    ax.scatter(xy[v1,1], xy[v1,0], c= jxy[v1,1]*(65/22)-32.5, s = 0.5, vmax=25*(65/22)-32.5, vmin=-32.5, cmap = 'gist_ncar', rasterized=True)
    if mouse_id == 1:
        xlims = ~(xy[:,1] > -1000) 
        ylims = ~(xy[:,0] < 450)
        selected_neurons = v1 * xlims * ylims
        ax.scatter(xy[selected_neurons,1], xy[selected_neurons,0], c= jxy[selected_neurons,1]*(65/22)-32.5, s = 0.5, vmax=25*(65/22)-32.5, vmin=0-32.5, cmap = 'gist_ncar', rasterized=True)
    if mouse_id == 5:
        lat = np.isin(iarea, [5]) 
        # ax.scatter(xy[lat,1], xy[lat,0], c= jxy[lat,0], s = 0.5, vmax=50, vmin=0, cmap = 'gist_ncar', rasterized=True)
        sc = ax.scatter(xy[lat,1], xy[lat,0], c= jxy[lat,1]*(65/22)-32.5, s = 0.5, vmax=25*(65/22)-32.5, vmin=0-32.5, cmap = 'gist_ncar', rasterized=True)
        ax.plot(out[8][:,1], out[8][:,0], '-k')
        ax.plot(out[5][:,1], out[5][:,0], '-k')
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.outline.set_visible(False)
        cbar.set_label('Elevation (deg)')
    else:
        ax.plot(out[8][:,1], out[8][:,0], '-k')
    rect0a = patches.Rectangle((-2750,400),742,636,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect0b = patches.Rectangle((-2620,1036),540,636,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect1a = patches.Rectangle((-2100,900),680,420,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect1b = patches.Rectangle((-2400,1320),680,420,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect3 = patches.Rectangle((-2450,500),672,412,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect4 = patches.Rectangle((-2550,700),680,417,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect5 = patches.Rectangle((-2700,800),664,414,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rect6 = patches.Rectangle((-1900,850),1426,578,linewidth=2,edgecolor='k',facecolor='none', zorder=2)
    rects0 = [rect0a,rect0b]
    rects1 = [rect1a,rect1b]
    rects = [rects0, rects1, rect3, rect4, rect5, rect6]
    if  mouse_id <= 1:
        r = rects[mouse_id]
        ax.add_patch(r[0])
        ax.add_patch(r[1])
    elif mouse_id > 1:
        ax.add_patch(rects[mouse_id])
    # ax.add_patch(rects[mouse_id])
    ax.set_aspect('equal')
    ax.set_axis_off()
    return il

def panelc_xretinotopy_model(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[2,mouse_id])
    pos = ax1.get_position().bounds
    ax1.axis('off')
    x_pixel_ratio = 0.75
    y_pixel_ratio = 0.5
    fev = data_dict['fev_all'][mouse_id]
    idx = np.where(fev>0.15)[0]
    xpos = data_dict['xpos_all'][mouse_id] / x_pixel_ratio
    ypos = data_dict['ypos_all'][mouse_id] / y_pixel_ratio 
    xlim_max = np.max(np.hstack(data_dict['xpos_all'])) / x_pixel_ratio
    xlim_min = np.min(np.hstack(data_dict['xpos_all'])) / x_pixel_ratio
    ylim_max = np.max(np.hstack(data_dict['ypos_all'])) / y_pixel_ratio
    ylim_min = np.min(np.hstack(data_dict['ypos_all'])) / y_pixel_ratio
    if mouse_id == 0:
        il = plot_label(ltr, il, ax1, transl, fs_title)
        ax1.set_title('Horizontal retinotopy (model)')
    if mouse_id == 1:
        idx_up = np.where(xpos>(325/x_pixel_ratio))[0]
        idx_down = np.where(xpos<=(325/x_pixel_ratio))[0]
        ymax = ypos[idx_up].max()
        xmax, xmin = xpos[idx_up].max(), xpos[idx_up].min()
        ypos[idx_up] = ymax - ypos[idx_up] + 300 # + ymax
    ylim_mid = (ylim_max + ylim_min) / 2
    xlim_mid = (xlim_max + xlim_min) / 2
    ymid = (xpos.max() + xpos.min()) / 2
    xmid = (ypos.max() + ypos.min()) / 2
    xpos = xpos + (ylim_mid - ymid)
    ypos = ypos + (xlim_mid - xmid) + 50
    ax = fig.add_axes([pos[0]+0.005, pos[1]+0.0, pos[2]-0.02, pos[3]-0.02])
    xpos_visual = data_dict['xpos_visual_all'][mouse_id]
    ypos_visual = data_dict['ypos_visual_all'][mouse_id]
    sc = ax.scatter(-ypos[idx], xpos[idx], c=xpos_visual[idx], s=1, cmap='gist_ncar', vmin=-135, vmax=(270 * (50/88))-135, rasterized=True)
    ax.set_title(f'mouse {mouse_id+1}', loc='center')
    ax.set_xlim(-ylim_max, -ylim_min)
    ax.set_ylim(xlim_min, xlim_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    return il

def paneld_yretinotopy_model(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[3,mouse_id])
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0], pos[1]-0.02, pos[2], pos[3]])
    ax1.axis('off')
    x_pixel_ratio = 0.75
    y_pixel_ratio = 0.5
    fev = data_dict['fev_all'][mouse_id]
    idx = np.where(fev>0.1)[0]
    xpos = data_dict['xpos_all'][mouse_id] / x_pixel_ratio
    ypos = data_dict['ypos_all'][mouse_id] / y_pixel_ratio 
    xlim_max = np.max(np.hstack(data_dict['xpos_all'])) / x_pixel_ratio
    xlim_min = np.min(np.hstack(data_dict['xpos_all'])) / x_pixel_ratio
    ylim_max = np.max(np.hstack(data_dict['ypos_all'])) / y_pixel_ratio
    ylim_min = np.min(np.hstack(data_dict['ypos_all'])) / y_pixel_ratio
    if mouse_id == 0:
        il = plot_label(ltr, il, ax1, transl, fs_title)
        ax1.set_title('Vertical retinotopy (model)')
    if mouse_id == 1:
        idx_up = np.where(xpos>325/x_pixel_ratio)[0]
        idx_down = np.where(xpos<=325/x_pixel_ratio)[0]
        ymax = ypos[idx_up].max()
        xmax, xmin = xpos[idx_up].max(), xpos[idx_up].min()
        ypos[idx_up] = ymax - ypos[idx_up] +300 # + ymax
    ylim_mid = (ylim_max + ylim_min) / 2
    xlim_mid = (xlim_max + xlim_min) / 2
    ymid = (xpos.max() + xpos.min()) / 2
    xmid = (ypos.max() + ypos.min()) / 2
    xpos = xpos + (ylim_mid - ymid)
    ypos = ypos + (xlim_mid - xmid) + 50
    xpos_visual = data_dict['xpos_visual_all'][mouse_id]
    ypos_visual = data_dict['ypos_visual_all'][mouse_id]
    ax = fig.add_axes([pos[0]+0.005, pos[1]-0.02, pos[2]-0.02, pos[3]-0.02])
    sc = ax.scatter(-ypos[idx], xpos[idx], c=ypos_visual[idx], s=1, cmap='gist_ncar', vmin=-32.5, vmax=65*(25/22)-32.5, rasterized=True)
    ax.set_title(f'mouse {mouse_id+1}', loc='center')
    ax.set_xlim(-ylim_max, -ylim_min)
    ax.set_ylim(xlim_min, xlim_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    return il


def panela_fev_distribution(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Mouse FEV distribution')
    ax = fig.add_axes([pos[0]+0.05, pos[1]-0.01, pos[2]-0.05, pos[3]-0.0])

    fev_all = data_dict['fev_all']
    ax.hist(fev_all, bins=20, color='gray', alpha=1, edgecolor='white', linewidth=2)
    ax.scatter(np.mean(fev_all), 4700, s=100, c='dimgray', marker='v', label='mean')
    ax.text(np.mean(fev_all)+0., 5000, f'{np.mean(fev_all):.2f}', ha='center', fontsize=10)
    ax.set_xlabel('FEV')
    ax.set_ylabel('Number of neurons')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 5000)
    ax.set_xlim(-0.05, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panelb_example_neuron1(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1:3])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.axis('off')
    ax1.set_title('Mouse example neurons')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]-0.0, pos[3]-0.0])

    spks_rep = data_dict['example_repeats']
    fev = data_dict['example_fev']
    # fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    idxes = np.arange(200,300)
    for i in range(3):
        spks_neuron = 0.07 * spks_rep[:,:,i] # / spks_rep[:,:,i].max()
        spks_neuron = spks_neuron[idxes]
        nstim, nrepeats = spks_neuron.shape
        for j in range(nrepeats):
            ax.plot(spks_neuron[:,j] + 1.1*i, color='k', alpha=0.25)
        ax.plot(spks_neuron.mean(axis=1) + 1.1*i, color='k', linewidth=1)
        ax.text(0.05, (0.8+1.1*i)/3.5, f'FEV = {fev[i]:.2f}', transform=ax.transAxes)
        ax.set_ylim([0,3.3])
        ax.set_axis_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return il

def panelc_monkey_fev_distribution(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Monkey FEV distribution')
    ax = fig.add_axes([pos[0]+0.05, pos[1]-0.01, pos[2]-0.05, pos[3]-0.0])
    fev_all = data_dict['monkey_fev_all']
    ax.hist(fev_all, bins=10, color='gray', alpha=1, edgecolor='white', linewidth=2)
    ax.scatter(np.mean(fev_all), 37, s=100, c='dimgray', marker='v', label='mean')
    ax.text(np.mean(fev_all), 40, f'{np.mean(fev_all):.2f}', ha='center', fontsize=10)
    ax.set_xlabel('FEV')
    ax.set_ylabel('Number of neurons')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 40)
    ax.set_xlim(0, 1)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def paneld_monkey_example_neuron1(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,1:3])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.axis('off')
    ax1.set_title('Monkey example neurons')
    pos = ax1.get_position().bounds
    ax = fig.add_axes([pos[0]+0.0, pos[1]-0.01, pos[2]-0.0, pos[3]-0.0])
    spks_rep = data_dict['monkey_example_repeats'].transpose(1,0,2)
    fev = data_dict['monkey_example_fev']
    idxes = np.arange(600,700)
    for i in range(3):
        spks_neuron = 0.1 * spks_rep[:,:,i] # 
        spks_neuron = spks_neuron[idxes]
        nstim, nrepeats = spks_neuron.shape
        for j in range(nrepeats):
            ax.plot(spks_neuron[:,j] + 1.1*i, color='k', alpha=0.3)
        ax.plot(spks_neuron.mean(axis=1) + 1.1*i, color='k', linewidth=1)
        ax.text(0.05, (0.8+1.1*i)/3.5, f'FEV = {fev[i]:.2f}', transform=ax.transAxes)
        ax.set_ylim([0,3.3])
        ax.set_axis_off()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return il


######################################################################## FIGURE 3 ########################################################################

def panela_wx(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse Wx')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.05, pos[2]-0.0, pos[3]-0.0])
    ax.set_title('mouse Wx', loc='center')
    interp_Wx_values = data_dict['interp_Wx_values']
    common_x = data_dict['common_x'] * (2*270/264)
    mean_Wx = np.nanmean(interp_Wx_values, axis=0)
    std_Wx = np.nanstd(interp_Wx_values, axis=0)
    # Plot the mean Wx, now properly aligned and interpolated
    ax.plot(common_x[5:-5], mean_Wx[5:-5], color='black', linewidth=2)
    ax.fill_between(common_x[5:-5], mean_Wx[5:-5]-std_Wx[5:-5], mean_Wx[5:-5]+std_Wx[5:-5], color='gray', alpha=0.5)

    ax.set_xlabel('visual angle (deg)')
    ax.set_ylabel('Wx')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.text(0.1, 0.9, f'FEV > 0.8', transform=ax.transAxes)
    ax.text(0.05, 0.9, f'NN={len(interp_Wx_values)}', transform=ax.transAxes)
    # ax.set_xlim(-55, 55)
    ax.set_ylim(-0.01, 0.2)
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  

    return il

def panelb_wy(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse Wy')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.05, pos[2]-0.0, pos[3]-0.0])
    ax.set_title('mouse Wy', loc='center')
    interp_Wy_values = data_dict['interp_Wy_values']
    common_x = data_dict['common_y'] * (2*270/264)
    mean_Wy = np.nanmean(interp_Wy_values, axis=0)
    std_Wy = np.nanstd(interp_Wy_values, axis=0)

    ax.plot(common_x[5:-5], mean_Wy[5:-5], color='black', linewidth=2)
    ax.fill_between(common_x[5:-5], mean_Wy[5:-5]-std_Wy[5:-5], mean_Wy[5:-5]+std_Wy[5:-5], color='gray', alpha=0.5)

    ax.set_xlabel('visual angle (deg)')
    ax.set_ylabel('Wy')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.text(0.1, 0.9, f'FEV > 0.8', transform=ax.transAxes)
    ax.text(0.05, 0.9, f'NN={len(interp_Wy_values)}', transform=ax.transAxes)
    # ax.set_xlim(-25, 25)
    ax.set_xticks([-40, 0, 40])
    ax.set_ylim(-0.01, 0.2)
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il


def panelc_monkey_wx(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('monkey Wx')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.05, pos[2]-0.0, pos[3]-0.0])
    ax.set_title('monkey Wx', loc='center')
    interp_Wx_values = data_dict['monkey_interp_Wx_values']
    common_x = data_dict['monkey_common_x'] * (2*1.1/80)
    mean_Wx = np.nanmean(interp_Wx_values, axis=0)
    std_Wx = np.nanstd(interp_Wx_values, axis=0)
    # Plot the mean Wx, now properly aligned and interpolated
    ax.plot(common_x[5:-5], mean_Wx[5:-5], color='black', linewidth=2)
    ax.fill_between(common_x[5:-5], mean_Wx[5:-5]-std_Wx[5:-5], mean_Wx[5:-5]+std_Wx[5:-5], color='gray', alpha=0.5)

    ax.set_xlabel('visual angle (deg)')
    ax.set_ylabel('Wx')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.text(0.1, 0.9, f'FEV > 0.8', transform=ax.transAxes)
    ax.text(0.05, 0.9, f'NN={len(interp_Wx_values)}', transform=ax.transAxes)
    # ax.set_xlim(-25, 25)
    ax.set_ylim(-0.01, 0.15)
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il

def paneld_monkey_wy(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,3])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse Wy')
    ax = fig.add_axes([pos[0]+0.04, pos[1]+0.05, pos[2]-0.0, pos[3]-0.0])
    ax.set_title('monkey Wy', loc='center')
    interp_Wy_values = data_dict['monkey_interp_Wy_values']
    common_x = data_dict['monkey_common_y'] * (2*1.1/80)
    mean_Wy = np.nanmean(interp_Wy_values, axis=0) 
    std_Wy = np.nanstd(interp_Wy_values, axis=0)

    ax.plot(common_x[5:-5], mean_Wy[5:-5], color='black', linewidth=2)
    ax.fill_between(common_x[5:-5], mean_Wy[5:-5]-std_Wy[5:-5], mean_Wy[5:-5]+std_Wy[5:-5], color='gray', alpha=0.5)

    ax.set_xlabel('visual angle (deg)')
    ax.set_ylabel('Wy')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.05, 0.9, f'NN={len(interp_Wy_values)}', transform=ax.transAxes)
    ax.set_ylim(-0.01, 0.15)
    ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il

######################################################################## FIGURE 4 ########################################################################

def panela_sparsity_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Validation result')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['mouse_feve_val']
    nconv2_all = data_dict['mouse_nconv2_val']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.03, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.27, 0.30)
    ax.set_yticks([0.27, 0.28, 0.29, 0.30])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panela_sparsity_5k_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse (5000 train images)', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['mouse_feve_val_5k']
    nconv2_all = data_dict['mouse_nconv2_val_5k']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.1, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.25, 0.27)
    ax.set_yticks([0.25, 0.26, 0.27])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panelb_monkey_sparsity_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('monkey', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['monkey_feve_val']
    nconv2_all = data_dict['monkey_nconv2_val']
    hs_list = data_dict['sparsity_monkey_hs']
    idx = np.arange(10)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.004, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il


def panela_sparsity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Test result')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_feve_all']
    nconv2_all = data_dict['sparsity_nconv2_all']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=0)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=0)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.03, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.62, 0.71)
    ax.set_yticks([0.62, 0.65, 0.68, 0.71])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panela_sparsity_5k(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse (5000 train images)', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_feve_5k_all']
    nconv2_all = data_dict['sparsity_nconv2_5k_all']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=0)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=0)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.1, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.52, 0.56)
    ax.set_yticks([0.52, 0.54, 0.56])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panelb_monkey_sparsity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,1])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('monkey')
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('monkey', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_monkey_feve_all']
    nconv2_all = data_dict['sparsity_monkey_nconv2_all']
    hs_list = data_dict['sparsity_monkey_hs']
    idx = np.arange(10)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=1)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=1)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.004, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.51, 0.57)
    ax.set_yticks([0.51, 0.53, 0.55, 0.57])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il

######################################################################## FIGURE 5 ########################################################################

def panela_fullmodel(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Fullmodel structure')
    ax1.axis('off')
    return il

def panelb_minimodel(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Minimodel structure')
    ax1.axis('off')
    return il

######################################################################## FIGURE 6 ########################################################################
def panela_reuse_conv1(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0:2,3:5])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.03, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    nmouse = 6
    ax1.set_title('Reuse conv1 performance')
    ax = fig.add_axes([pos[0]+0.09, pos[1]+0.0, pos[2]-0.1, pos[3]-0.0])
    feve_matrix = data_dict['reuse_conv1_feve'] 
    im = ax.imshow(feve_matrix.mean(2))
    cbar = plt.colorbar(im, ax=ax, ticks = [0.67, 0.72, 0.77], fraction=0.01, pad=0.04) #  ticks = [65, 71, 77]
    # cbar.ax.set_yticklabels(['54%', '60%', '66%'])
    cbar.set_label('FEVE')
    # set cbar value range
    im.set_clim(0.67, 0.77)
    # set cbar outline invisible
    cbar.outline.set_visible(False)
    ax.set_xlabel('Performance on', fontsize = 14)
    ax.set_ylabel('Trained with conv1 from', fontsize = 14)
    ax.set_xticks(np.arange(nmouse), [f'mouse {i+1}' for i in range(nmouse)], rotation=30, ha='right')
    ax.set_yticks(np.arange(nmouse), [f'mouse {i+1}' for i in range(nmouse)])
    for spine in ax.spines.values():
        spine.set_visible(False)
    mean_feve = feve_matrix.mean(2)
    num_rows, num_cols = mean_feve.shape
    for i in range(num_rows):
        for j in range(num_cols):
            ax.text(j, i, f'{mean_feve[i, j]:.2f}', ha='center', va='center', color='w', fontweight='bold')
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  
    return il


def panelb_conv1_kernels(fig, grid, data_dict, il, transl, imouse):
    ax1 = plt.subplot(grid[imouse//3, imouse%3])
    if imouse == 0:
        il = plot_label(ltr, il, ax1, transl, fs_title)
        ax1.set_title('Conv1 weights')
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.02, pos[1]-0.0, pos[2]-0.02, pos[3]-0.0])
    # ax = fig.add_axes([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    conv1_W = data_dict['conv1_W'][imouse]
    # ax1.set_position([pos[0]+0.0, pos[1]-0.01, pos[2]-0.0, pos[3]-0.0])
    from matplotlib import gridspec
    inner = gridspec.GridSpecFromSubplotSpec(4,4,
                        subplot_spec=grid[imouse//3, imouse%3], wspace=0.2, hspace=0.0)
    # ax1.remove()
    isort_all = [[13,5,6,15,10,12,2,12,1,3,9,14,7,0,11,4], 
            [0,13,15,8, 14,2,11,9,7,5,1,4,6,12,3,10],
            [10,9,4,12,6,7,14,11,10,3,5,2,13,15,0,1],
            [14,3,0,4,5,12,11,10,6,7,8,9,13,15,1,2],
            [11,3,15,10,8,6,12,2,13,5,4,7,14,1,0,9],
            [0,9,12,1,5,7,6,15,14,13,3,8,10,11,4,2]]
    isort = isort_all[imouse]
    for i in range(16):
        ax = fig.add_subplot(inner[i//4, i%4])
        im = ax.imshow(conv1_W[isort[i]], cmap='RdBu', vmin=-0.2, vmax=0.2)
        ax.axis('off')
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        yadd = (i//4) * 0.005
        new_pos = [pos.x0+0.02, pos.y0 - 0.04 + yadd, pos.width-0.00, pos.height]  # Adjust the 0.05 as needed
        ax.set_position(new_pos)
    pos = ax.get_position()
    ax = fig.add_subplot(grid[imouse//3, imouse%3])    
    ax.set_title(f'mouse {imouse+1}', x=0.46, y=0.86)
    ax.axis('off')
    return il

######################################################################## FIGURE 7 ########################################################################

def panela_high_catvar(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Mouse high category variance neurons')
    ax1.axis('off')
    return il

def panelb_low_catvar(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Mouse low category variance neurons')
    ax1.axis('off')
    return il

def panela_high_catvar_monkey(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Monkey high category variance neurons')
    ax1.axis('off')
    return il

def panelb_low_catvar_monkey(fig, grid, data_dict, il, transl, mouse_id):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('Monkey low category variance neurons')
    ax1.axis('off')
    return il


def test_channel_output(model, img_test, batch_size=100):
    model.eval()
    n_test = img_test.shape[0]
    conv2_features = []
    conv2_indepth_features = []
    with torch.no_grad():
        for k in np.arange(0, n_test, batch_size):
            kend = min(k+batch_size, n_test)
            img_batch = img_test[k:kend]


            x = model.core.features.layer0(img_batch)
            conv2_indepth_fv = model.core.features.layer1.ds_conv.in_depth_conv(x)
            x = model.core.features.layer1.ds_conv.spatial_conv(conv2_indepth_fv)
            x = model.core.features.layer1.norm(x)
            conv2_relu_fvs = model.core.features.layer1.activation(x)
            # print('after in_depth_conv: ', conv2_indepth_fv.shape, conv2_indepth_fv.max(), conv2_indepth_fv.min())
            # print('after conv2_relu: ', conv2_relu_fvs.shape, conv2_relu_fvs.max(), conv2_relu_fvs.min())
            conv2_fv = conv2_relu_fvs[:, :, 16, 32]
            conv2_indepth_fv = conv2_indepth_fv[:, :, 16, 32]
            conv2_features.append(conv2_fv.detach().cpu().numpy())
            conv2_indepth_features.append(conv2_indepth_fv.detach().cpu().numpy())
            # spks_test_pred[k:kend] = spks_pred
    conv2_features = np.vstack(conv2_features)
    conv2_indepth_features = np.vstack(conv2_indepth_features)
    return conv2_indepth_features, conv2_features

from matplotlib import patches
def add_channel_frame(axs, row, col_start, col_end, color, alpha):
    ax = axs[row, col_start]  # Leftmost axis in the row
    # Rectangle coordinates (x, y) and dimensions (width, height)
    rect = patches.Rectangle(
        (-0.025, -0.05), (col_end - col_start + 1)*1.33, 1.1, transform=ax.transAxes,
        color=color, fill=False, linewidth=3, zorder=10, alpha=alpha,
        clip_on=False  # To ensure it draws outside the axes
    )
    ax.add_patch(rect)

######################################################################## Functions sparsity ########################################################################

def panela_sparsity_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Minimodel validation result')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['mouse_feve_val']
    nconv2_all = data_dict['mouse_nconv2_val']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.03, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.27, 0.30)
    ax.set_yticks([0.27, 0.28, 0.29, 0.30])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panela_sparsity_5k_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse (5000 train images)', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['mouse_feve_val_5k']
    nconv2_all = data_dict['mouse_nconv2_val_5k']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.1, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.25, 0.27)
    ax.set_yticks([0.25, 0.26, 0.27])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panelb_monkey_sparsity_val(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,1])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('monkey')
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('monkey', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['monkey_feve_val']
    nconv2_all = data_dict['monkey_nconv2_val']
    hs_list = data_dict['sparsity_monkey_hs']
    idx = np.arange(10)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.004, color='gray', linestyle='--')
    ax.axhline(y=0.99*feve_all[0], color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    # ax.set_ylim(0.51, 0.57)
    # ax.set_yticks([0.51, 0.53, 0.55, 0.57])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il


def panela_sparsity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('Minimodel test result')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_feve_all']
    nconv2_all = data_dict['sparsity_nconv2_all']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=0)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=0)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.03, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.62, 0.71)
    ax.set_yticks([0.62, 0.65, 0.68, 0.71])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panela_sparsity_5k(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('mouse')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('mouse (5000 train images)', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_feve_5k_all']
    nconv2_all = data_dict['sparsity_nconv2_5k_all']
    hs_list = data_dict['sparsity_hs']
    idx = np.arange(9)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=0)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=0)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.1, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.52, 0.56)
    ax.set_yticks([0.52, 0.54, 0.56])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    return il

def panelb_monkey_sparsity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,1])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    # ax1.set_title('monkey')
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    ax.set_title('monkey', loc='center')
    ax2 = ax.twinx()
    feve_all = data_dict['sparsity_monkey_feve_all']
    nconv2_all = data_dict['sparsity_monkey_nconv2_all']
    hs_list = data_dict['sparsity_monkey_hs']
    idx = np.arange(10)
    hs_list = np.array(hs_list)
    ax.plot(hs_list[idx], feve_all.mean(axis=1)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], nconv2_all.mean(axis=1)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax2.set_ylabel('# of conv2', color='b')
    ax.axvline(x=0.004, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    ax.set_ylim(0.51, 0.57)
    ax.set_yticks([0.51, 0.53, 0.55, 0.57])
    ax2.set_ylim(0, 65)
    ax2.set_yticks([0, 16, 32, 48, 64])
    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')  
    return il

def panelc_fullmodel_sparsity(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('16-320 model result')
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    # ax.set_title('mouse', loc='center')
    ax2 = ax.twinx()
    hs_list = data_dict['fullmodel_hoyer_hs']
    idx = np.arange(len(hs_list))
    hs_list = np.array(hs_list)
    feve_all = data_dict['fullmodel_hoyer_feve_all']
    n_wc_all = data_dict['fullmodel_hoyer_n_wc_all']
    ax.plot(hs_list[idx], feve_all.mean(axis=0)[idx], color='r', marker='o', markersize=3)
    ax2.plot(hs_list[idx], n_wc_all.mean(axis=0)[idx], color='b', marker='o', markersize=3)
    # set the axis as the same color as the line
    ax.yaxis.label.set_color('r')
    ax2.yaxis.label.set_color('b')
    ax.set_xlabel('Hoyer-Square regularization strength')
    ax.set_ylabel('FEVE', color='r')
    ax.set_ylim([0.60, 0.72])
    ax2.set_ylim([0, 320])
    ax2.set_ylabel('# of conv2', color='b')
    # ax.axvline(x=0.03, color='gray', linestyle='--')
    ax2.spines['right'].set_visible(True)
    # Color the ax1 y-axis spine, ticks and tick labels
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='red')
    # Color the ax2 y-axis spine, ticks and tick labels
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='blue')
    return il


def panelc_wc_visualization(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0, :2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    ax1.set_title('16-320 model Wc')
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.0, pos[1]-0.0, pos[2]-0.02, pos[3]-0.0])
    # ax2 = fig.add_axes([pos[0]-0.0, pos[1]-0.02, pos[2]-0.0, pos[3]+0.0])
    # conv1_W = data_dict['conv1_W'][imouse]
    # # ax1.set_position([pos[0]+0.0, pos[1]-0.01, pos[2]-0.0, pos[3]-0.0])
    wc_all = data_dict['fullmodel_hoyer_wc_all']
    hs_list = data_dict['fullmodel_hoyer_hs']
    feve_all = data_dict['fullmodel_hoyer_feve_all']
    n_wc_all = data_dict['fullmodel_hoyer_n_wc_all']
    from matplotlib import gridspec
    inner = gridspec.GridSpecFromSubplotSpec(2, 3,
                        subplot_spec=grid[0, :2], wspace=0.2, hspace=0.4)
    hs_selected =   [0,1,2,5,6,7]
    for ihs in range(6):
        ax = fig.add_subplot(inner[ihs//3, ihs%3])
        ireal = hs_selected[ihs]
        wc = wc_all[ihs]
        np.random.seed(0)
        ineurons = np.random.choice(wc.shape[0], 500, replace=False)
        for ineuron in ineurons:
            tmp = wc[ineuron]
            # normalize positive and negative weights separately
            pos = tmp[tmp > 0]
            pos = pos / np.max(pos)
            neg = tmp[tmp < 0]
            neg = neg / np.abs(np.min(neg))
            tmp[tmp > 0] = pos
            tmp[tmp < 0] = neg
            # tmp = tmp / np.max(np.abs(tmp))
            ax.plot(np.sort(tmp), label=f'neuron {ineuron}')
        # ax.set_title(f'HS={hs_list[ireal]}; FEVE={feve_all.mean(axis=0)[ireal]:.3f}; #conv2={n_wc_all.mean(axis=0)[ireal]:.0f}', fontsize=12, loc='center')
        ax.text(0.1, 0.6, f'HS={hs_list[ireal]}\nFEVE={feve_all[3][ireal]:.3f}\n#conv2={n_wc_all[3][ireal]:.0f}', transform=ax.transAxes, ha="left", fontsize=10)
        ax.set_xlabel('Wc index')
        ax.set_ylabel('Wc value')
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(-5, 320)
        ax.set_xticks([0, 160, 320])
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        new_pos = [pos.x0+0.03 - (ihs%3)*0.02, pos.y0 - 0.01*(ihs//3) - 0.01 , pos.width-0.04, pos.height - 0.0]  # Adjust the 0.05 as needed
        ax.set_position(new_pos)
        # ax.set_aspect(0.55/ax.get_data_ratio(), adjustable='box')
    return il

######################################################################## Functions same core ########################################################################

def panela_same_core_1layer(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE of 1-layer models')
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.0, pos[2]-0.0, pos[3]-0.02])
    # ax.set_title('mouse Wx', loc='center')
    
    
    feve_lurz = data_dict['feve_lurz_model'][:, 0]
    feve_ds = data_dict['same_core_1layer_feve_ds']
    feve_our = data_dict['same_core_1layer_feve_our']
    c2 = '#4cb938'
    nmouse = 6
    ax.scatter(np.zeros(nmouse), feve_lurz, color=c2, label='Lurz, 1 layer', alpha=0.5, s=10)
    ax.scatter(np.ones(nmouse), feve_ds, color='gray', label='our, 1 layer', alpha=0.5, s=10)
    ax.scatter(np.ones(nmouse)*2, feve_our, color='black', label='our, 1 layer', alpha=0.5, s=10)
    ax.errorbar(0, feve_lurz.mean(), yerr=feve_lurz.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(1, feve_ds.mean(), yerr=feve_ds.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(2, feve_our.mean(), yerr=feve_our.std(), fmt='o', color='black', alpha=0.5)                                                                                                                                                                                                                                                                                                                                                                                     
    for i in range(nmouse):
        ax.plot([0, 1], [feve_lurz[i], feve_ds[i]], color='black', alpha=0.2)
        ax.plot([1, 2], [feve_ds[i], feve_our[i]], color='black', alpha=0.2)
    # ax.text(1, 0.2, "Lurz, 1 layer", transform=ax.transAxes, ha="right", color=c2)
    # ax.text(1, 0.1, "our (same core), 1 layer", transform=ax.transAxes, ha="right", color='gray')
    # ax.text(1, 0.3, "our, 1 layer", transform=ax.transAxes, ha="right", color='black')
    ax.text(0.18+0.04, 0.85, f'{feve_lurz.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color=c2)
    ax.text(0.58+0.04, 0.85, f'{feve_ds.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='gray')
    ax.text(0.98+0.04, 0.85, f'{feve_our.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='black')
    # ax.set_xlabel('model', fontsize=14)
    ax.set_ylabel('Variance explained (FEVE)', fontsize=12)
    ax.set_ylim(0., 0.8)
    ax.set_xticks([0, 1, 2], ['sensorium core\n+\nsensorium readout', 'sensorium core\n+\nour readout', 'our core\n+\nour readout'], rotation=35)
    ax.set_yticks(np.arange(0., 0.81, 0.2), labels=[f'{i:.2f}' for i in np.arange(0., 0.81, 0.2)])
    ax.set_xlim(-0.2, 2.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title('FEVE with the same core')
    # plt.tight_layout()
    print(feve_lurz.mean(), feve_ds.mean())
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')  

    return il

def panelb_same_core_2layer(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE of 2-layer models')
    ax1.set_position([pos[0]+0.02, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.04, pos[1]-0.0, pos[2]-0.0, pos[3]-0.02])
    # ax.set_title('

    feve_lurz = data_dict['feve_lurz_model'][:, 1]
    feve_ds = data_dict['same_core_2layer_feve_ds']
    feve_our = data_dict['same_core_2layer_feve_our']
    c2 = '#4cb938'
    nmouse = 6
    ax.scatter(np.zeros(nmouse), feve_lurz, color=c2, label='Lurz, 2 layer', alpha=0.5, s=10)
    ax.scatter(np.ones(nmouse), feve_ds, color='gray', label='our (same core), 2 layer', alpha=0.5, s=10)
    ax.scatter(2*np.ones(nmouse), feve_our, color='gray', label='our, 2 layer', alpha=0.5, s=10)
    ax.errorbar(0, feve_lurz.mean(), yerr=feve_lurz.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(1, feve_ds.mean(), yerr=feve_ds.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(2, feve_our.mean(), yerr=feve_our.std(), fmt='o', color='black', alpha=0.5)
    for i in range(nmouse):
        ax.plot([0, 1], [feve_lurz[i], feve_ds[i]], color='black', alpha=0.2)
        ax.plot([1, 2], [feve_ds[i], feve_our[i]], color='black', alpha=0.2)
    # ax.text(1, 0.3, "Lurz, 2 layer", transform=ax.transAxes, ha="right", color=c2)
    # ax.text(1, 0.2, "our (same core), 2 layer", transform=ax.transAxes, ha="right", color='gray')
    # ax.text(1, 0.1, "our (192-192), 2 layer", transform=ax.transAxes, ha="right", color='gray')
    ax.text(0.18+0.04, 0.6, f'{feve_lurz.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color=c2)
    ax.text(0.58+0.04, 0.6, f'{feve_ds.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='gray')
    ax.text(0.98+0.04, 0.6, f'{feve_our.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='gray')
    # ax.set_xlabel('model', fontsize=14)
    ax.set_ylabel('Variance explained (FEVE)', fontsize=12)
    ax.set_ylim(0., 0.8)
    ax.set_xticks([0, 1, 2], ['sensorium core\n+\nsensorium readout', 'sensorium core\n+\nour readout', 'our core\n+\nour readout'], rotation=35)
    ax.set_yticks(np.arange(0., 0.81, 0.2), labels=[f'{i:.2f}' for i in np.arange(0., 0.81, 0.2)])
    ax.set_xlim(-0.2, 2.2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_title('FEVE with the same core')
    # plt.tight_layout()
    print(feve_lurz.mean(), feve_ds.mean())
    ax.set_aspect(1/ax.get_data_ratio(), adjustable='box')  
    
    return il

def panelc_kernel_size(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.set_title('FEVE with different kernel sizes')
    ax1.axis('off')
    # ax1.set_title('mouse Wx')
    ax1.set_position([pos[0]+0.03, pos[1]+0.0, pos[2]-0.06, pos[3]-0.0])
    ax = fig.add_axes([pos[0]-0.01, pos[1]+0.025, pos[2]-0.0, pos[3]-0.05])
    
    conv1_ks_list = [7,13,17,21,25,29]
    conv2_ks_list = [5,7,9,11,13,15]
    
    feve_conv_ks_all = data_dict['same_core_conv_ks_feve']
    feve_conv_ks = np.mean(feve_conv_ks_all, axis=0)
    import seaborn as sns
    mycmap = sns.color_palette("mako", as_cmap=True)
    im = ax.imshow(feve_conv_ks, cmap=mycmap, vmin=0.65, vmax=0.75)
    # im = ax.imshow(feve_conv_ks, cmap='viridis', vmin=0.5, vmax=0.74)
    ax.set_xlabel('conv2 kernel size', fontsize=14)
    ax.set_ylabel('conv1 kernel size', fontsize=14)
    ax.set_xticks(np.arange(len(conv2_ks_list)), labels=conv2_ks_list)
    ax.set_yticks(np.arange(len(conv1_ks_list)), labels=conv1_ks_list)
    # ax.set_title('FEVE with different kernel sizes')
    # colorbar set to be small
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('FEVE')

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')  
    
    return il


######################################################################## Functions pool no pool ########################################################################

def panela_pool_nopool(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('Fullmodel structure')
    ax1.axis('off')
    ax1.set_title('FEVE of 1-layer models')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]+0.05, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.05, pos[1]+0.03, pos[2]-0.01, pos[3]-0.06])
    # ax.set_title('mouse Wx', loc='center')

    feve_pool = data_dict['feve_pool']
    feve_no_pool = data_dict['feve_no_pool']
    feve_small_ks = data_dict['feve_small_ks']
    nmouse = 6
    # fig, ax = plt.subplots(1, 1, figsize=(3,3))
    ax.scatter(np.zeros(nmouse), feve_pool, color='gray', label='16-320, pool', alpha=0.5, s=10)
    ax.scatter(np.ones(nmouse), feve_no_pool, color='purple', label='16-320, no pool', alpha=0.5, s=10)
    ax.scatter(2*np.ones(nmouse), feve_small_ks, color='darkblue', label='16-320, small ks', alpha=0.5, s=10)
    ax.errorbar(0, feve_pool.mean(), yerr=feve_pool.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(1, feve_no_pool.mean(), yerr=feve_no_pool.std(), fmt='o', color='black', alpha=0.5)
    ax.errorbar(2, feve_small_ks.mean(), yerr=feve_small_ks.std(), fmt='o', color='darkblue', alpha=0.5)
    for i in range(nmouse):
        ax.plot([0, 1, 2], [feve_pool[i], feve_no_pool[i], feve_small_ks[i]], color='black', alpha=0.2)
    ax.text(1, 0.3, "16-320, pool", transform=ax.transAxes, ha="right", color='gray')
    ax.text(1, 0.2, "16-320, no pool", transform=ax.transAxes, ha="right", color='purple')
    ax.text(1, 0.1, "16-320, small ks", transform=ax.transAxes, ha="right", color='darkblue')
    ax.text(0.25, 0.6, f'{feve_pool.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='gray')
    ax.text(0.62, 0.6, f'{feve_no_pool.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='purple')
    ax.text(1.05, 0.6, f'{feve_small_ks.mean()*100:.1f}%', transform=ax.transAxes, ha="right", color='darkblue')
    ax.set_xlabel('model', fontsize=14)
    ax.set_ylabel('Variance explained (FEVE)', fontsize=12)
    ax.set_ylim(0., 0.8)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks([0, 1, 2], ['pool', 'no pool', 'small ks'])
    ax.set_yticks(np.arange(0., 0.81, 0.2), labels=[f'{i:.2f}' for i in np.arange(0., 0.81, 0.2)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1.2/ax.get_data_ratio(), adjustable='box')  

    return il

def panelb_nopool_kernels(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    # ax1.set_title('Minimodel structure')
    ax1.axis('off')
    pos = ax1.get_position().bounds
    ax1.set_position([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('Conv1 weights of the model without pooling')
    # ax = fig.add_axes([pos[0]-0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    mouse_id = 5
    conv1_W = data_dict['conv1_no_pool'][mouse_id]
    # ax1.set_position([pos[0]+0.0, pos[1]-0.01, pos[2]-0.0, pos[3]-0.0])
    from matplotlib import gridspec
    inner = gridspec.GridSpecFromSubplotSpec(4,4,
                        subplot_spec=grid[0,1], wspace=0.2, hspace=0.0)
    # ax1.remove()
    isort = [9,0,12,1,7,5,6,14,15,13,10,3,8,11,4,2]
    for i in range(16):
        ax = fig.add_subplot(inner[i//4, i%4])
        im = ax.imshow(conv1_W[isort[i]], cmap='RdBu', vmin=-0.2, vmax=0.2)
        ax.axis('off')
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        yadd = (i//4) * 0.015
        xadd = (i%4) * 0.04
        new_pos = [pos.x0+0.02-xadd, pos.y0 - 0.0 - yadd, pos.width-0.003, pos.height]  # Adjust the 0.05 as needed
        ax.set_position(new_pos)
    
    return il

######################################################################## Functions vary nneuron ########################################################################
def panel_vary_nneuron(fig, grid, data_dict, il, transl, i):
    ax1 = plt.subplot(grid[0,i])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.03, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    nmouse = 6
    titles = ['model trained with 5k images', 
              'model trained with 30k images \n(pretrained conv1)', 
              'model trained with 30k images',
              'model trained with 30k images \n(pretrained conv1)']
    ax1.set_title(titles[i], loc='left', y=1-(i%2)*0.08)
    ax = fig.add_axes([pos[0]+0.07, pos[1]+0.03, pos[2]-0.07, pos[3]-0.11])
    nstim_list = [5000, 30000]
    if i%2 == 0: feves_all = data_dict['nstim_feve_all']
    else: feves_all = data_dict['nstim_feve_pretrain_all']
    k = i//2
    nstims = nstim_list[k]
    feves = feves_all[k].reshape(2, -1)
    ax.scatter(np.zeros(feves.shape[1]), feves[0], color='k', s=10, alpha=0.5)
    ax.scatter(np.ones(feves.shape[1]), feves[1], color='k', s=10, alpha=0.5)
    for i in range(feves.shape[1]):
        ax.plot([0,1], [feves[0,i], feves[1,i]], 'k', lw=0.5, alpha=0.1)
    # plot the mean and connect them
    ax.plot([0], [feves[0].mean()], 'ro')
    ax.plot([1], [feves[1].mean()], 'ro')
    ax.plot([0,1], [feves[0].mean(), feves[1].mean()], 'r', lw=1)
    ax.set_xticks([0,1], [f'1 neuron', 'all neurons'])
    ax.set_xlim(-0.5, 1.5)
    # set right and top spines invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #  signifiance
    from scipy.stats import wilcoxon
    _, p = wilcoxon(feves[0], feves[1])
    if p < 0.001:
        ax.text(0.5, 0.98, '***', transform=ax.transAxes, ha='center', fontsize=12)
    elif p < 0.01:
        ax.text(0.5, 0.98, '**', transform=ax.transAxes, ha='center', fontsize=12)
    elif p < 0.05:
        ax.text(0.5, 0.98, '*', transform=ax.transAxes, ha='center', fontsize=12)
    else:
        ax.text(0.5, 0.98, 'n.s.', transform=ax.transAxes, ha='center', fontsize=12)
    ax.text(0.14, feves[0].mean()+0.0, f'{feves[0].mean():.3f}', transform=ax.transAxes, ha='center', va='center')
    ax.text(0.91, feves[1].mean()+0.0, f'{feves[1].mean():.3f}', transform=ax.transAxes, ha='center', va='center')
    ax.set_ylabel('FEVE')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 1.5)
    # ax.set_aspect(1.4/ax.get_data_ratio(), adjustable='box')  
    return il


######################################################################## Functions conv2 cluster ########################################################################

def panela_conv2_tsne(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('t-SNE of conv2 1x1 weights')
    ax = fig.add_axes([pos[0]-0.05, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    X_2d = data_dict['tsne_conv2_1x1']
    mouse_ids_flat = data_dict['mouse_ids_flat']
    cmap = plt.get_cmap('Set3')
    for i in range(nmouse):
        idxes = np.where(mouse_ids_flat == i)[0]
        ax.scatter(X_2d[idxes, 0], X_2d[idxes, 1], color=cmap(i), label=f'mouse {i+1}', alpha=0.2, s=2, rasterized=True)
    # ax.legend()
        ax.text(1.0, 0.65-i*0.07, f'mouse {i+1}', transform=ax.transAxes, ha="left", color=cmap(i))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # ax.set_title('t-SNE of 1x1 conv2 weights (normalize by the norm)')
    ax.set_axis_off()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # plt.tight_layout()

    # load mouse 0 cov1 weights
    return il

def panela_conv2_tsne_cluster(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('t-SNE of conv2 spatial weights')
    ax = fig.add_axes([pos[0]-0.05, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    X_2d = data_dict['tsne_conv2_1x1']
    cluster_labels = data_dict['cluster_labels']
    cluster_center_samples = data_dict['center_cluster_samples']
    cluster_center_idxes = data_dict['cluster_center_idxes']
    n_clusters = len(cluster_center_samples)
    cmap = plt.get_cmap('viridis', n_clusters+1)
    for i in range(n_clusters):
        idxes = np.where(cluster_labels == i)[0]
        ax.scatter(X_2d[idxes, 0], X_2d[idxes, 1], color=cmap(i), label=f'cluster {i+1}', alpha=0.05, s=5, rasterized=True)
        ax.text(1.05, 0.7-i*0.07, f'cluster {i+1}', transform=ax.transAxes, ha="left", color=cmap(i), fontsize=12, alpha=0.7)
    for i in range(len(cluster_center_samples)):
        idx = cluster_center_idxes[i]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c='gray', marker='x', s=100, label=f'Cluster {i}')
    ax.text(0.95, 0.7 - nmouse * 0.07, f'X', transform=ax.transAxes, ha="left", color='gray', fontsize=12)
    ax.text(0.99, 0.7 - nmouse * 0.07, f'Cluster Centers', transform=ax.transAxes, ha="left", color='gray', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # ax.set_title('t-SNE of 1x1 conv2 weights (normalize by the norm)')
    ax.set_axis_off()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # plt.tight_layout()
    return il

def panela_conv2_visualization(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]-0.0, pos[1]-0.0, pos[2]-0.05, pos[3]-0.0])
    ax1.set_title('Visualize cluster centers')
    # ax = fig.add_axes([pos[0]-0.2, pos[1]-0.04, pos[2]-0.0, pos[3]-0.0])
    conv1_W = data_dict['conv1_W']
    ori_cluster_centers = data_dict['ori_cluster_centers']
    from matplotlib import colors as mcolors
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.bwr
    # cmap_frame = plt.get_cmap('tab10')
    # ncluster = len(isort_cluster)
    # Visualize the conv1 weights
    n_clusters = ori_cluster_centers.shape[0]
    cmap_frame= plt.get_cmap('viridis', n_clusters+1)
    n_top = 8
    from matplotlib import gridspec
    inner = gridspec.GridSpecFromSubplotSpec(n_clusters, n_top,
                        subplot_spec=grid[0,2], wspace=0.0, hspace=0.01)
    for i in range(n_clusters):
        cluster_w = ori_cluster_centers[i]
        idxes = np.argsort(np.abs(cluster_w))[::-1][:n_top] # keep the top 8 channels
        isort = np.argsort(cluster_w[idxes])[::-1]
        isort = idxes[isort]
        for j in range(n_top):
            ax = fig.add_subplot(inner[i, j])
            ax.imshow(conv1_W[isort[j]], cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            ax.axis('off')

            # Create the frame color
            frame_color = cmap(norm(cluster_w[isort[j]]))
            alpha = abs(cluster_w[isort[j]])
            # Add a rectangle frame around the image
            rect = plt.Rectangle(
                (-0.5, -0.5),  # Starting point
                conv1_W[isort[j]].shape[1],  # Width
                conv1_W[isort[j]].shape[0],  # Height
                linewidth=5, edgecolor=frame_color[:3] + (alpha,), facecolor='none'
            )       
            ax.add_patch(rect)
        # add_channel_frame(ax, i, 0, 1.027*ncluster, cmap_frame(i), 1)
            if j == 0:
                ax.text(-0.15, 0.4, f'Cluster {i+1}', transform=ax.transAxes, ha="right", color=cmap_frame(i))
            pos = ax.get_position()
            # Adjust the bottom position to move subplot down
            new_pos = [pos.x0+0.06-j*0.005, pos.y0+i*0.005 , pos.width-0.008, pos.height - 0.008] 
            ax.set_position(new_pos)

    return il


def panelb_conv2_spatial_tsne(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('t-SNE of conv2 spatial weights')
    ax = fig.add_axes([pos[0]-0.05, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    X_2d = data_dict['tsne_conv2_spatial']
    mouse_ids_flat = data_dict['mouse_ids_flat']
    cmap = plt.get_cmap('Set3')
    for i in range(nmouse):
        idxes = np.where(mouse_ids_flat == i)[0]
        ax.scatter(X_2d[idxes, 0], X_2d[idxes, 1], color=cmap(i), label=f'mouse {i+1}', alpha=0.2, s=2, rasterized=True)
    # ax.legend()
        ax.text(1.0, 0.65-i*0.07, f'mouse {i+1}', transform=ax.transAxes, ha="left", color=cmap(i))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # ax.set_title('t-SNE of 1x1 conv2 weights (normalize by the norm)')
    ax.set_axis_off()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    # plt.tight_layout()

    # load mouse 0 cov1 weights
    return il

def panelb_conv2_spatial_tsne_cluster(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('t-SNE of conv2 spatial weights')
    ax = fig.add_axes([pos[0]-0.05, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    X_2d = data_dict['tsne_conv2_spatial']
    cluster_labels = data_dict['spatial_cluster_labels']
    cluster_center_samples = data_dict['spatial_center_cluster_samples']
    cluster_center_idxes = data_dict['spatial_cluster_center_idxes']
    n_clusters = len(cluster_center_samples)
    cmap = plt.get_cmap('viridis', n_clusters+1)
    for i in range(n_clusters):
        idxes = np.where(cluster_labels == i)[0]
        ax.scatter(X_2d[idxes, 0], X_2d[idxes, 1], color=cmap(i), label=f'cluster {i+1}', alpha=0.05, s=5, rasterized=True)
        ax.text(1.15+0.1, 0.7-i*0.07, f'cluster {i+1}', transform=ax.transAxes, ha="right", color=cmap(i), fontsize=12, alpha=0.7)
    for i in range(len(cluster_center_samples)):
        idx = cluster_center_idxes[i]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c='gray', marker='x', s=100, label=f'Cluster {i}')
    ax.text(0.95+0.0, 0.7 - nmouse * 0.07, f'X', transform=ax.transAxes, ha="left", color='gray', fontsize=12)
    ax.text(0.99+0.0, 0.7 - nmouse * 0.07, f'Cluster Centers', transform=ax.transAxes, ha="left", color='gray', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # ax.set_title('t-SNE of 1x1 conv2 weights (normalize by the norm)')
    ax.set_axis_off()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panelb_conv2_spatial_visualization(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[1,2])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('Visualize cluster centers')
    from matplotlib import colors as mcolors
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    # Visualize the conv1 weights
    ori_cluster_centers = data_dict['spatial_ori_cluster_centers']
    # isort_cluster = data_dict['spatial_isort_cluster']
    n_clusters = ori_cluster_centers.shape[0]
    cmap = plt.get_cmap('viridis', n_clusters+1)
    n_top = 8
    from matplotlib import gridspec
    ncols = int(n_clusters/2)
    inner = gridspec.GridSpecFromSubplotSpec(2, ncols, 
                        subplot_spec=grid[1,2], wspace=0.1, hspace=0.1)
    
    for i in range(n_clusters):
        ax = fig.add_subplot(inner[i//ncols, i%ncols])
        spatial_w = ori_cluster_centers[i]
        spatial_w = spatial_w / np.linalg.norm(spatial_w)
        # smooth the spatial weights
        import cv2
        spatial_w = cv2.GaussianBlur(spatial_w, (3, 3), 0)
        ax.imshow(spatial_w, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        ax.axis('off')

        # Create the frame color
        frame_color = cmap(i)
        # Add a rectangle frame around the image
        rect = plt.Rectangle(
            (-0.5, -0.5),  # Starting point
            spatial_w.shape[1],  # Width
            spatial_w.shape[0],  # Height
            linewidth=10, edgecolor=frame_color[:3], facecolor='none'
        )
        ax.add_patch(rect)
        pos = ax.get_position()
        # Adjust the bottom position to move subplot down
        new_pos = [pos.x0+0.05-0.01*(i%ncols), pos.y0 + 0.04*(i//ncols)-0.02, pos.width-0.02, pos.height - 0.00] 
        ax.set_position(new_pos)

    return il


def panelc_conv2_channel_tsne(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    nmouse = 6
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('t-SNE of conv2 channel activities')
    ax = fig.add_axes([pos[0]-0.05, pos[1]-0.0, pos[2]-0.0, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(6,6))
    cmap = plt.get_cmap('Set3')
    mouse_ids_flat = data_dict['mouse_ids_flat']
    X_tsne = data_dict['tsne_conv2_channel']
    for i in range(nmouse):
        idxes = np.where(mouse_ids_flat == i)[0]
        ax.scatter(X_tsne[idxes, 0], X_tsne[idxes, 1], color=cmap(i), label=f'mouse {i+1}', alpha=0.1, s=5, rasterized=True)
        ax.text(1., 0.65-i*0.07, f'mouse {i+1}', transform=ax.transAxes, ha="left", color=cmap(i))
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    # ax.set_title('t-SNE of conv2 channel responses')
    ax.set_axis_off()
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    return il

def panelc_conv2_channel_rastermap(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.0, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax1.set_title('Rastermap of conv2 channel activities')
    ax = fig.add_axes([pos[0]+0.02, pos[1]+0.005, pos[2]+0.0, pos[3]-0.02])
    x = data_dict['rastermap_x']
    im = ax.imshow(x[:, :1000], aspect='auto', cmap='gray_r', vmin=0, vmax=2)
    ax.set_xlabel('Stimuli')
    ax.set_ylabel('Channels')
    # set left and bottom spines invisible
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # # set ticks invisible
    ax.set_xticks([])
    ax.set_yticks([])
    # add colorbar below the rastermap
    # cbar outside the rastermap
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, orientation='horizontal', aspect=10)

    cbar.set_label('z-scored', fontsize=10)
    # set location of the colorbar to the bottom left of the rastermap
    pos = ax.get_position()
    cbar.ax.set_position([pos.x0+0.0, pos.y0-0.21, pos.width/10, pos.height-0.01])
    # set ticks of the colorbar
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['0', '1', '2'], fontsize=8)

    # add a box on the rastermap, with y from 280 to 290, and x from 600 to 700
    rect = plt.Rectangle((600, 280), 100, 5, linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    return il

def panelc_conv2_channel_rastermap_zoomin(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[2,2])
    # il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_position([pos[0]+0.09, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    # ax1.set_title('Rastermap of conv2 channel activities (binned)')
    ax = fig.add_axes([pos[0]+0.06, pos[1]+0.03, pos[2]-0.05, pos[3]-0.05])
    channel_resp = data_dict['rastermap_channel_resp']
    isort = data_dict['rastermap_isort']
    y = channel_resp[isort]
    start = 39*280 +100
    ax.imshow(y[start:start+100, 600:700], aspect='auto', cmap='gray_r', vmin=0, vmax=2)   
    # ax.set_xlabel('Stimuli')
    # ax.set_ylabel('Channels')
    # set left and bottom spines invisible
    # set axes invisible
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # add a rect outside the rastermap
    rect = plt.Rectangle((0, 0), 99, 99, linewidth=5, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    return il

######################################################################## Functions gabor ########################################################################

def panela_feve_fev(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE of our model and FEV')                                   
    ax = fig.add_axes([pos[0]+0.03, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(3,3))
    fev = data_dict['fev_gabor']
    feve_fullmodel = data_dict['feve_fullmodel_gabor']
    ax.scatter(fev, feve_fullmodel, s=1, alpha=0.1, color='gray', rasterized=True)
    ax.set_xlabel('FEV')
    ax.set_ylabel('FEVE')
    from scipy.stats import pearsonr
    r, p = pearsonr(fev, feve_fullmodel)
    ax.text(0.7, 0.15, f'r={r:.2f}\np<0.001', transform=ax.transAxes, ha="left", color='k')
    return il


def panelb_feve_distribution(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE distribution')
    ax = fig.add_axes([pos[0]+0.04, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    fixed_bar_width = 0.05  # Set a fixed bar width for all distributions
    feve_minimodel = data_dict['feve_fullmodel_gabor']
    feve_gabor = data_dict['feve_gabor']

    bins = np.arange(-1, 1, fixed_bar_width)
    ax.hist(feve_minimodel, bins=bins, alpha=0.5, label='fullmodel', color='gray', histtype='step')
    ax.hist(feve_gabor, bins=bins, alpha=0.5, label='Gabor', color='green', histtype='step')
    ax.set_xlabel('FEVE')
    ax.set_ylabel('# of neurons')
    ax.text(0.1, 0.9, f'our model', color='gray', transform=ax.transAxes, ha='left')
    ax.text(0.1, 0.8, f'Gabor model', color='green', transform=ax.transAxes, ha="left") 
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, 2800])
    ax.set_yticks(np.arange(0, 2801, 400))
    # ax.set_yticks(np.arange(0, 5), [f'{i*fixed_bar_width:.2f}' for i in np.arange(5)])
    return il

def panelc_simple_complex(fig, grid, data_dict, il, transl):
    ax1 = plt.subplot(grid[0,1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    ax1.set_title('FEVE distribution')
    ax1.set_position([pos[0]-0.02, pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    ax = fig.add_axes([pos[0]+0.02, pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])
    # fig, ax = plt.subplots(1, 1, figsize=(3,3))
    cratio = data_dict['cratio']

    feve_gabor = data_dict['feve_gabor']
    ivalid = np.where(feve_gabor> 0.)[0]
    icomplex = np.where(cratio > 0.5)[0]
    isimple = np.where(cratio <= 0.5)[0]
    icomplex = np.intersect1d(icomplex, ivalid)
    isimple = np.intersect1d(isimple, ivalid)       
    fullmodel_feve = data_dict['feve_fullmodel_gabor']

    fixed_bar_width = 0.05  # Set a fixed bar width for all distributions
    bins = np.arange(0, 1, fixed_bar_width)
    ax.hist(fullmodel_feve[icomplex], bins=bins, alpha=1, label='complex', color='r', histtype='step')
    ax.text(0.05, 0.9, f'complex cell', transform=ax.transAxes, color='r')
    ax.hist(fullmodel_feve[isimple], bins=bins, alpha=1, label='simple', color='b', histtype='step')
    ax.text(0.05, 0.8, f'simple cell', transform=ax.transAxes, color='b')

    ax.set_xlabel('FEVE')
    ax.set_ylabel('# of neurons')
    # set right spine invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_yticks(np.arange(0, 5), [f'{i*fixed_bar_width:.2f}' for i in np.arange(5)])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 2000])
    return il

def paneld_gabor_params(fig, grid, data_dict, il, transl, k):
    ax1 = plt.subplot(grid[0,k+1])
    il = plot_label(ltr, il, ax1, transl, fs_title)
    pos = ax1.get_position().bounds
    ax1.axis('off')
    names = ['', 'Gabor filter spatial frequency (f)', 'Gabor filter size (\u03C3)', 'Gabor filter orientation (\u03B8)']
    ax1.set_title(f'{names[k]}')
    xpos_shift = [0, 0.04, 0.01, 0.02]
    ax1.set_position([pos[0]-xpos_shift[k], pos[1]+0.0, pos[2]-0.0, pos[3]-0.0])
    xpos_shift = [0,-0.01, 0.01, 0]
    ax = fig.add_axes([pos[0]+0.02+xpos_shift[k], pos[1]-0.02, pos[2]-0.08, pos[3]-0.0])

    feve_gabor = data_dict['feve_gabor']
    ivalid = np.where(feve_gabor> 0.)[0]
    fullmodel_feve = data_dict['feve_fullmodel_gabor'][ivalid]
    cratio = data_dict['cratio'][ivalid]
    mf = data_dict['mf'][ivalid]
    msigma = data_dict['msigma'][ivalid]
    mtheta = data_dict['mtheta'][ivalid]

    params = [cratio, mf, msigma, mtheta]
    titles = ['c2/(c1+c2)', 'f', '\u03C3', '\u03B8']
    from fractions import Fraction
    param = params[k]
    val, cts = np.unique(param, return_counts=True)
    #fig, ax = plt.subplots(1, 1, figsize=(4,4))
    #ax.set_title(f'{titles[k]}', ha='center')       
    fixed_bar_width = 0.05  # Set a fixed bar width for all distributions
    cmap = plt.cm.get_cmap('viridis', len(val) + 1)
    # Iterate over the unique values in the parameter array
    for i, v in enumerate(val):
        # Get data for the current parameter value
        data = fullmodel_feve[param == v]
        
        # Calculate bins to ensure consistent bar width
        min_data, max_data = np.min(data), np.max(data)
        # print(min_data, max_data)
        bins = np.arange(min_data, max_data + fixed_bar_width, fixed_bar_width)
        
        # Plot histogram with the fixed bar width
        ax.hist(data, bins=bins, alpha=0.7, label=f'{titles[k]}={v:.1f}', 
                histtype='step', color=cmap(i))
        
        # Add information text
        if k == 3:
            denom = v / np.pi
            denom = Fraction(denom).limit_denominator()
            str_label = f'{denom} π'
            if v == 0: str_label = '0'
            ax.text(0.1, 0.9 - i * 0.09, 
                    str_label, 
                    transform=ax.transAxes, color=cmap(i))
        else:
            ax.text(0.1, 0.9 - i * 0.09, 
                    f'{v:.2f}', 
                    transform=ax.transAxes, color=cmap(i))
    ms = [0,6,5,5]
    m = ms[k]
    # ax.set_yticks(np.arange(m), [f'{i*fixed_bar_width:.2f}' for i in np.arange(m)])
    ax.set_xlabel('FEVE')
    ax.set_ylabel('# of neurons')
    # set right spine invisible
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0, 1])
    ymax = [0, 1200, 700, 500]
    ax.set_ylim([0, ymax[k]])
    # set yticks
    if k == 1:
        ax.set_yticks(np.arange(0, ymax[k]+1, 200))
    else:
        ax.set_yticks(np.arange(0, ymax[k]+1, 100))
    # plt.legend()
    return il

#################################### FIGURES ####################################

def figure1(dat, root):
    fig = plt.figure(figsize=(14,10))
    grid = plt.GridSpec(4,6, figure=fig, left=0.03, right=0.95, top=0.96, bottom=0.05, 
                        wspace = 0.1, hspace = 0.1, height_ratios=[5, 5, 4, 4])
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    for mouse_id in range(6):
        il = panela_xretinotopy(fig, grid, dat, il, transl, mouse_id)
    for mouse_id in range(6):
        il = panelb_yretinotopy(fig, grid, dat, il, transl, mouse_id)
    for mouse_id in range(6):
        il = panelc_xretinotopy_model(fig, grid, dat, il, transl, mouse_id)
    for mouse_id in range(6):
        il = paneld_yretinotopy_model(fig, grid, dat, il, transl, mouse_id)

    fig.savefig(os.path.join(root, 'sup_fig1.pdf'), dpi=200) 

def figure2(dat, root):
    fig = plt.figure(figsize=(14,7))
    grid = plt.GridSpec(2,3, figure=fig, left=0.1, right=0.9, top=0.96, bottom=0.05, 
                        wspace = 0.3, hspace = 0.2)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_fev_distribution(fig, grid, dat, il, transl)
    il = panelb_example_neuron1(fig, grid, dat, il, transl)
    il = panelc_monkey_fev_distribution(fig, grid, dat, il, transl)
    il = paneld_monkey_example_neuron1(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig2.pdf'), dpi=200) 


def figure3(dat, root):
    fig = plt.figure(figsize=(14,3))
    grid = plt.GridSpec(1,4, figure=fig, left=0.03, right=0.95, top=0.9, bottom=0.05, 
                        wspace = 0.3, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_wx(fig, grid, dat, il, transl)
    il = panelb_wy(fig, grid, dat, il, transl)
    il = panelc_monkey_wx(fig, grid, dat, il, transl)
    il = paneld_monkey_wy(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig3.pdf'), dpi=200) 

def figure4(dat, root):
    fig = plt.figure(figsize=(14,12)) # fig = plt.figure(figsize=(14,8))
    grid = plt.GridSpec(3,3, figure=fig, left=0.04, right=0.99, top=0.97, bottom=0.07, 
                        wspace = 0.1, hspace = 0.4)
    # grid = plt.GridSpec(2,3, figure=fig, left=0.02, right=0.99, top=0.9, bottom=0.16, wspace = 0.1, hspace = 0.4)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panelc_wc_visualization(fig, grid, dat, il, transl)
    il = panelc_fullmodel_sparsity(fig, grid, dat, il, transl)
    il = panela_sparsity_val(fig, grid, dat, il, transl)
    il = panelb_monkey_sparsity_val(fig, grid, dat, il, transl)
    il = panela_sparsity_5k_val(fig, grid, dat, il, transl)
    il = panela_sparsity(fig, grid, dat, il, transl)
    il = panelb_monkey_sparsity(fig, grid, dat, il, transl)
    il = panela_sparsity_5k(fig, grid, dat, il, transl)
    
    fig.savefig(os.path.join(root, 'sup_fig4_revision.pdf'), dpi=200) 

def figure5(dat, root):
    fig = plt.figure(figsize=(14,14))
    grid = plt.GridSpec(2,1, figure=fig, left=0.18, right=0.75, top=0.9, bottom=0.16, 
                        wspace = 0.1, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_fullmodel(fig, grid, dat, il, transl, 0)
    il = panelb_minimodel(fig, grid, dat, il, transl, 0)

    fig.savefig(os.path.join(root, 'sup_fig5.pdf'), dpi=200) 

def figure6(dat, root):
    fig = plt.figure(figsize=(14,7))
    grid = plt.GridSpec(2,5, figure=fig, left=0.02, right=0.95, top=0.9, bottom=0.16, 
                        wspace = 0.2, hspace = 0.15)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    for imouse in range(6):
        il = panelb_conv1_kernels(fig, grid, dat, il, transl, imouse)
    il = panela_reuse_conv1(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig6.pdf'), dpi=200) 


def figure8(dat, root):
    fig = plt.figure(figsize=(14,14))
    grid = plt.GridSpec(2,1, figure=fig, left=0.45, right=0.75, top=0.9, bottom=0.16, 
                        wspace = 0.1, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_high_catvar(fig, grid, dat, il, transl, 0)
    il = panelb_low_catvar(fig, grid, dat, il, transl, 0)

    fig.savefig(os.path.join(root, 'sup_fig8.pdf'), dpi=200) 

def figure9(dat, root):
    fig = plt.figure(figsize=(14,14))
    grid = plt.GridSpec(2,1, figure=fig, left=0.45, right=0.75, top=0.9, bottom=0.16, 
                        wspace = 0.1, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_high_catvar_monkey(fig, grid, dat, il, transl, 0)
    il = panelb_low_catvar_monkey(fig, grid, dat, il, transl, 0)

    fig.savefig(os.path.join(root, 'sup_fig9.pdf'), dpi=200) 

def figure_gabor(dat, root):
    fig = plt.figure(figsize=(14,3))
    grid = plt.GridSpec(1,5, figure=fig, left=0.02, right=0.99, top=0.8, bottom=0.25, 
                        wspace = 0.1, hspace = 0.1)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panelb_feve_distribution(fig, grid, dat, il, transl)
    il = panelc_simple_complex(fig, grid, dat, il, transl)
    for k in [1,2,3]:
        il = paneld_gabor_params(fig, grid, dat, il, transl, k)

    fig.savefig(os.path.join(root, 'sup_fig_gabor.pdf'), dpi=200) 

def figure_same_core(dat, root):
    fig = plt.figure(figsize=(14,4))
    grid = plt.GridSpec(1,3, figure=fig, left=0.1, right=0.9, top=0.92, bottom=0.33, 
                        wspace = 0.3, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_same_core_1layer(fig, grid, dat, il, transl)
    il = panelb_same_core_2layer(fig, grid, dat, il, transl)
    il = panelc_kernel_size(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig_same_core_revision.pdf'), dpi=200) 

def figure_pool_nopool(dat, root):
    fig = plt.figure(figsize=(14,3))
    grid = plt.GridSpec(1,2, figure=fig, left=0.15, right=0.95, top=0.9, bottom=0.05, 
                        wspace = 0.3, hspace = 0.3)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_pool_nopool(fig, grid, dat, il, transl)
    il = panelb_nopool_kernels(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig_pool_nopool.pdf'), dpi=200) 

def figure_vary_nneuron(dat, root):
    fig = plt.figure(figsize=(14,3))
    grid = plt.GridSpec(1,4, figure=fig, left=0.02, right=0.95, top=0.9, bottom=0.05, 
                        wspace = 0.2, hspace = 0.15)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    for i in range(4):
        il = panel_vary_nneuron(fig, grid, dat, il, transl, i)

    fig.savefig(os.path.join(root, 'sup_fig_vary_nneuron.pdf'), dpi=200) 

def figure_conv2_cluster(dat, root):
    fig = plt.figure(figsize=(14,12))
    grid = plt.GridSpec(3,3, figure=fig, left=0.05, right=0.95, top=0.95, bottom=0.1, 
                        wspace = 0.1, hspace = 0.2)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    # conv2 1x1
    il = panela_conv2_tsne(fig, grid, dat, il, transl)
    il = panela_conv2_tsne_cluster(fig, grid, dat, il, transl)
    il = panela_conv2_visualization(fig, grid, dat, il, transl)

    # conv2 spatial
    il = panelb_conv2_spatial_tsne(fig, grid, dat, il, transl)
    il = panelb_conv2_spatial_tsne_cluster(fig, grid, dat, il, transl)
    il = panelb_conv2_spatial_visualization(fig, grid, dat, il, transl)

    # # conv2 channel activity
    il = panelc_conv2_channel_tsne(fig, grid, dat, il, transl)
    il = panelc_conv2_channel_rastermap(fig, grid, dat, il, transl)
    il = panelc_conv2_channel_rastermap_zoomin(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig_conv2_cluster.pdf'), dpi=200) 

