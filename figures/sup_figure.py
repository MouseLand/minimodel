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
    fig = plt.figure(figsize=(14,8))
    grid = plt.GridSpec(2,3, figure=fig, left=0.02, right=0.99, top=0.9, bottom=0.16, 
                        wspace = 0.1, hspace = 0.4)
    
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    il = 0
    il = panela_sparsity_val(fig, grid, dat, il, transl)
    il = panelb_monkey_sparsity_val(fig, grid, dat, il, transl)
    il = panela_sparsity_5k_val(fig, grid, dat, il, transl)
    il = panela_sparsity(fig, grid, dat, il, transl)
    il = panelb_monkey_sparsity(fig, grid, dat, il, transl)
    il = panela_sparsity_5k(fig, grid, dat, il, transl)

    fig.savefig(os.path.join(root, 'sup_fig4.pdf'), dpi=200) 

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