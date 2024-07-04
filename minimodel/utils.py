
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def weight_bandwidth(w, return_peak=False):
    '''
    Calculate the bandwidth of the weights.

    Parameters:
    ----------
    w : numpy.ndarray
        1D array of weights.
    return_peak : bool, optional
        If True, also return the center position of the peak. Default is False.

    Returns:
    -------
    bandwidth : float
        The bandwidth of the weights.
    centerpos : float, optional
        The center position of the peak. Returned only if `return_peak` is True.
    '''
    # interpolate the weights
    x = np.arange(len(w))
    nx = len(w)
    nxnew = 2*nx
    xnew = np.linspace(0, len(w)-1, nxnew)
    w = np.interp(xnew, x, w)
    # smooth the weights
    w = gaussian_filter(w, sigma=3, truncate=10)
    # w = np.sign(w[np.argmax(np.abs(w))]) * w
    # find the peak of the weights
    peak_idx = np.argmax(w)
    # find the left and right half width at half maximum, which is closest to the peak
    left_idx = np.argmin(np.abs(w[:peak_idx] - w[peak_idx]/2)) if peak_idx > 0 else 0
    # find the closest left_idx to the peak_idx
    if isinstance(left_idx, np.ndarray):
        left_idx = left_idx[np.argmin(np.abs(left_idx - peak_idx))]
    right_idx = (np.argmin(np.abs(w[peak_idx:] - w[peak_idx]/2)) + peak_idx) if peak_idx < len(w) else len(w)
    # find the closest right_idx to the peak_idx
    if isinstance(right_idx, np.ndarray):
        right_idx = right_idx[np.argmin(np.abs(right_idx - peak_idx))]
    # calculate the bandwidth
    bandwidth = (right_idx - left_idx) * nx / nxnew
    centerpos = (right_idx + left_idx) * nx / nxnew / 2
    if return_peak:
        return bandwidth, centerpos
    return bandwidth

def get_image_mask(model, Ly=66, Lx=130):
    '''
     Generate an image mask based on the model's readout weights.

    Parameters:
    ----------
    model : torch.nn.Module
        The neural network model with readout weights.
    Ly : int, optional
        Height of the output mask. Default is 66.
    Lx : int, optional
        Width of the output mask. Default is 130.

    Returns:
    -------
    ineuron_mask_up : numpy.ndarray
        The generated image mask, upsampled and adjusted to fit the image size.
    '''

    Wc = model.readout.Wc.detach().cpu().numpy().squeeze()
    # # change model Wx and Wy
    Wx = model.readout.Wx.detach().cpu().numpy()
    Wy = model.readout.Wy.detach().cpu().numpy()
    # outer product of Wx and Wy
    Wxy = np.einsum('icj,ick->ijk', Wy, Wx).squeeze()

    # rfsize from the Wxy
    NN = 1
    bandwidth_Wx = np.zeros(NN)
    bandwidth_Wy = np.zeros(NN)
    centerX = np.zeros(NN)
    centerY = np.zeros(NN)
    for i in range(NN):
        bandwidth_Wx[i], centerX[i] = weight_bandwidth(Wx[i, 0, :], return_peak=True)
        bandwidth_Wy[i], centerY[i] = weight_bandwidth(Wy[i, 0, :], return_peak=True)
    rf_size = bandwidth_Wx * bandwidth_Wy
    print(f'rf size: {np.mean(rf_size):.2f}')

    import cv2
    ineuron_mask = Wxy / np.abs(Wxy).max()

    Ymax, Xmax = ineuron_mask.shape
    # print(Ymax, Xmax)
    bandwidth_ratio = 1
    # cut the mask based on bandwidth and center
    xstart = int(centerX-bandwidth_Wx//bandwidth_ratio) 
    xend = int(centerX+bandwidth_Wx//bandwidth_ratio+1)
    ystart = int(centerY-bandwidth_Wy//bandwidth_ratio)
    yend = int(centerY+bandwidth_Wy//bandwidth_ratio+1)
    if xstart < 0:
        xstart = 0
    if xend > Xmax:
        xend = Xmax
    if ystart < 0:
        ystart = 0
    if yend > Ymax:
        yend = Ymax
    cutted_mask = ineuron_mask[ystart:yend, xstart:xend]
    # print(cutted_mask.shape, xstart, xend, ystart, yend)
    # upsample the mask
    adjust_pixel = 25
    cutted_mask_up = cv2.resize(cutted_mask, (cutted_mask.shape[1]*2 + adjust_pixel, cutted_mask.shape[0]*2 + adjust_pixel))

    print('cutted_mask_up: ', cutted_mask_up.shape)
    cutted_mask_up = (cutted_mask_up - cutted_mask_up.min()) / (cutted_mask_up.max() - cutted_mask_up.min())
    # add a eclipse mask on the cutted mask
    mask = np.zeros_like(cutted_mask_up)
    mask = cv2.ellipse(mask, (cutted_mask_up.shape[1]//2, cutted_mask_up.shape[0]//2), (cutted_mask_up.shape[1]//2, cutted_mask_up.shape[0]//2), 0, 0, 360, 1, -1)
    cutted_mask_up = cutted_mask_up * mask
    cutted_mask_up_y, cutted_mask_up_x = cutted_mask_up.shape
    new_xstart = int(centerX*2 - cutted_mask_up_x//2)
    new_xend = int(centerX*2 + cutted_mask_up_x//2)+1 
    new_ystart = int(centerY*2 - cutted_mask_up_y//2)
    new_yend = int(centerY*2 + cutted_mask_up_y//2)+1 
    new_yend += cutted_mask_up_y - (new_yend - new_ystart)
    new_xend += cutted_mask_up_x - (new_xend - new_xstart)
    print(new_xstart, new_xend, new_ystart, new_yend, cutted_mask_up.shape)


    Ymax, Xmax = Ly, Lx
    ineuron_mask_up = np.zeros((Ymax, Xmax))
    # adjust the edge
    if new_xstart < 0:
        cutted_mask_up = cutted_mask_up[:, -new_xstart:]
        new_xstart = 0
        new_xend = cutted_mask_up.shape[1]
        print('adjusting xstart')
    if new_xend > Xmax:
        cutted_mask_up = cutted_mask_up[:, :-(new_xend-Xmax)]
        new_xend = Xmax
        print('adjusting xend')
    if new_ystart < 0:
        cutted_mask_up = cutted_mask_up[-new_ystart:, :]
        new_ystart = 0
        new_yend = cutted_mask_up.shape[0]
        print('adjusting ystart')
    if new_yend > Ymax:
        cutted_mask_up = cutted_mask_up[:-(new_yend-Ymax), :]
        # print(cutted_mask_up.shape)
        new_yend = Ymax
        print('adjusting yend')
    ineuron_mask_up[new_ystart:new_yend, new_xstart:new_xend] = cutted_mask_up
    return ineuron_mask_up

# Function to add a frame around a channel
from matplotlib import patches
def add_channel_frame(axs, row, col_start, col_end, color, alpha, monkey=False):
    """
    Add a frame around a specified range of columns in a subplot grid.

    Parameters:
    ----------
    axs : matplotlib.axes._subplots.AxesSubplot
        The array of subplot axes.
    row : int
        The row index of the subplot grid where the frame should be added.
    col_start : int
        The starting column index of the frame.
    col_end : int
        The ending column index of the frame.
    color : str
        The color of the frame.
    alpha : float
        The transparency level of the frame (0 to 1).
    monkey : bool, optional
        Adjusts the frame size for monkey data. Default is False.
    
    Returns:
    -------
    None
    """
    ax = axs[row, col_start]  # Leftmost axis in the row
    if monkey: adjust_value = 1.64
    else: adjust_value = 1.33
    # Rectangle coordinates (x, y) and dimensions (width, height)
    rect = patches.Rectangle(
        (-0.025, -0.05), (col_end - col_start + 1)*adjust_value , 1.1, transform=ax.transAxes,
        color=color, fill=False, linewidth=3, zorder=10, alpha=alpha,
        clip_on=False  # To ensure it draws outside the axes
    )
    ax.add_patch(rect)
