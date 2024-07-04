import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class Core(nn.Module):
    """ core convolutional layers.
    depth separable conv structure from https://github.com/sinzlab/neuralpredictors 
    """
    def __init__(self, n_channels, kernel_size, stride, out_channels=None, dense=False, downsample=3, conv_init=None, conv_all=False, \
        depth_separable=False, pool=False, activation='elu', \
        lessconv=False, batchnorm=True, avgpool=False, spatial_nconv=False):
        super().__init__()
        self.features = nn.Sequential()
        self.dense = dense
        self.downsample = downsample
        self.conv_all = conv_all # if True, convolve with all layer features, otherwise only convolve with last layer feature
        self.pool = pool
        self.lessconv = lessconv
        self.activation = activation
        self.avgpool = avgpool
        self.spatial_nconv = spatial_nconv
        if (len(n_channels) > 2):
            self.add_layers(n_channels, kernel_size, stride, out_channels, separable=depth_separable, batchnorm=batchnorm)
        self.apply(self.init_weights)
        # initialize conv0 with one-layer model
        if conv_init is not None:
            W = conv_init
            if kernel_size[0] != W.shape[-1]:
                import cv2
                Wup = np.array([cv2.resize(Wi, (kernel_size[0], kernel_size[0])) for Wi in W])
            else:
                Wup = W
            Wup = torch.from_numpy(Wup).unsqueeze(1)
            self.features.layer0.conv.weight.data = Wup

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
    
    def add_layers(self, in_channels, kernel_size, stride, out_channels=None, separable=False, batchnorm=True):
        if out_channels is None:
            out_channels = in_channels[1:]
        for i in range(len(in_channels)-1):
            layer = OrderedDict()
            if (i==0) or (not separable): # first layer is not depth separable
                layer['conv'] = nn.Conv2d(in_channels[i], out_channels[i],
                                          kernel_size[i], 
                                          padding=kernel_size[i]//2, 
                                          stride=stride if i==0 else 1, 
                                          bias=False)
                
                if batchnorm:
                    layer['norm'] = nn.BatchNorm2d(out_channels[i], momentum=0.9, 
                                                eps=1e-5, affine=True, track_running_stats=True)
                if self.activation == 'elu':
                    layer['activation'] = nn.ELU()
                elif self.activation == 'relu':
                    layer['activation'] = nn.ReLU()
                if self.pool:
                    if self.avgpool:
                        layer['pool'] = nn.AvgPool2d(2)
                    else:
                        layer['pool'] = nn.MaxPool2d(2)
            else:
                if not self.spatial_nconv: spatial_nconv = out_channels[i]
                else: spatial_nconv = self.spatial_nconv    
                layer['ds_conv'] = nn.Sequential()
                layer['ds_conv'].add_module('in_depth_conv', nn.Conv2d(in_channels[i], spatial_nconv, 1, bias=False))
                layer['ds_conv'].add_module('spatial_conv', nn.Conv2d(spatial_nconv, spatial_nconv,
                                                                      kernel_size[i], padding=kernel_size[i]//2,
                                                                      groups=spatial_nconv, bias=False))
                if not self.lessconv:
                    layer['ds_conv'].add_module('out_depth_conv', nn.Conv2d(spatial_nconv, out_channels[i], 1, bias=False))
                if batchnorm:
                    layer['norm'] = nn.BatchNorm2d(out_channels[i], momentum=0.9, 
                                                eps=1e-5, affine=True, track_running_stats=True)
                if self.activation == 'elu':
                    layer['activation'] = nn.ELU()
                elif self.activation == 'relu':
                    layer['activation'] = nn.ReLU()
            self.features.add_module(f'layer{i}', nn.Sequential(layer))
            
    def forward(self, img):
        features = self.features[0](img)
        all_features = [features]
        for n in range(1, len(self.features)):
            new_features = self.features[n](features)
            if self.dense:
                # features = torch.cat((features, new_features), axis=1)\
                features = new_features + features
            else:
                features = new_features
            all_features.append(features)
        if self.conv_all:
            all_features = torch.cat(all_features, axis=1)
            features = all_features
        return features
    
    def orth_reg(self):
        in_W = self.features[1].ds_conv.in_depth_conv.weight
        in_W = in_W.reshape(in_W.size(0), -1)
        spatial_W = self.features[1].ds_conv.spatial_conv.weight
        spatial_W = spatial_W.reshape(spatial_W.size(0), -1)
        in_corr = in_W @ in_W.t()
        spatial_corr = spatial_W @ spatial_W.t()
        mask = (in_corr > 0) & (spatial_corr > 0)
        loss = torch.abs(spatial_corr - torch.eye(spatial_corr.size(0), device=spatial_corr.device)) * mask
        return loss.sum()


class Readout(nn.Module):
    def __init__(self, in_shape, n_neurons, y_init=None, x_init=None, c_init=None, coef_init=None, 
    rank=1, yx_separable=True, bias_init=None, poisson=False, activation='elu', Wc_coef=0.01, Wxy_gabor_init=0.1, Wxy_init=0.01):
        super().__init__()
        self.yx_separable = yx_separable
        n_conv, Ly, Lx = in_shape
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        if self.yx_separable:
            Wy = Wxy_init * torch.randn((n_neurons, rank, Ly))
            Wx = Wxy_init * torch.randn((n_neurons, rank, Lx))       
            if y_init is not None: 
                Wy[np.arange(n_neurons), :, y_init] += Wxy_gabor_init
            if x_init is not None: 
                Wx[np.arange(n_neurons), :, x_init] += Wxy_gabor_init
            
            self.Wy = nn.Parameter(Wy)
            self.Wx = nn.Parameter(Wx)
        else:
            Wyx = .01 * torch.randn((n_neurons, Ly, Lx))
            Wyx[np.arange(n_neurons), y_init, x_init] += Wxy_gabor_init
            self.Wyx = nn.Parameter(Wyx)
            
        Wc = Wc_coef * torch.randn((n_neurons, rank, n_conv))
        if (c_init is not None) and (n_neurons > 1):
            Wc[np.arange(n_neurons), 0, c_init] += 0.5*torch.from_numpy(coef_init.astype('float32'))

        self.Wc = nn.Parameter(Wc)
        if rank > 1:
            self.bias = nn.Parameter(torch.from_numpy(bias_init.astype('float32'))) if bias_init is not None else nn.Parameter(torch.zeros((rank, n_neurons)))
        else:
            self.bias = nn.Parameter(torch.from_numpy(bias_init.astype('float32'))) if bias_init is not None else nn.Parameter(torch.zeros(n_neurons))
        self.use_poisson = poisson
        self.rank = rank
    
    def forward(self, conv):
        if self.yx_separable:
            if self.rank > 1:
                pred = torch.einsum('nrc, nky, icyx, nkx->irn ', self.Wc, self.Wy, conv, self.Wx)
            else:
                pred = torch.einsum('nrc, nry, icyx, nrx->in ', self.Wc, self.Wy, conv, self.Wx)

            pred = pred.add(self.bias)
            pred = self.activation(pred)

            if self.rank > 1:
                pred = pred.sum(axis=1)
        else:
            pred = torch.einsum('nc, icyx, nyx->in', self.Wc, conv, self.Wyx)
        return pred
        
    def weight_regularizer(self):
        return 0
    
    def l1_norm(self):
        Wc_l1 = self.Wc.abs().sum(axis=(1,2))
        return Wc_l1
    
    def l2_norm(self):
        Wc_l2 = (self.Wc**2).sum(axis=(1,2))
        Wy_l2 = (self.Wy**2).sum(axis=(1,2))
        Wx_l2 = (self.Wx**2).sum(axis=(1,2))
        return Wc_l2 + Wy_l2 + Wx_l2
    
    def hoyer_square(self):
        wc_l1 = self.Wc.abs().sum(axis=(1,2))
        wc_l2 = (self.Wc**2).sum(axis=(1,2))
        return wc_l1**2 / wc_l2


class Encoder(nn.Module):
    def __init__(self, core, readout, loss_fun='poisson', device=torch.device('cuda')):
        super().__init__()
        self.core = core
        self.readout = readout
        self.mean_img = 128
        self.std_img = 61
        self.loss_fun = loss_fun
        self.bias = 1e-12
        self.bceloss = nn.BCELoss(reduction='none')
        self.device = device

    def forward(self, img, detach_core=False):
        x = self.core(img)
        if detach_core:
            x = x.detach()
        x = self.readout(x)
        x += 1 + self.bias
        return x
        
    def loss_function(self, spks_batch, spks_pred, l1_readout=0, l2_readout=0, hs_reg=0.0):
        loss = (spks_pred - spks_batch * torch.log(spks_pred)).sum(axis=0)
        loss += l1_readout * self.readout.l1_norm()
        loss += l2_readout * self.readout.l2_norm()
        loss += hs_reg * self.readout.hoyer_square()
        loss = loss.mean()
        return loss

    def responses(self, images, core=False, batch_size=8, device=torch.device('cuda')):
        nimg = images.shape[0]
        n_batches = int(np.ceil(nimg/batch_size))
        self.eval()
        for k in range(n_batches):
            inds = np.arange(k*batch_size, min(nimg, (k+1)*batch_size))
            # data = torch.from_numpy((images[inds] - self.mean_img) / self.std_img).to(device)
            data = torch.from_numpy(images[inds]).to(device)
            data = data.unsqueeze(1)
            with torch.no_grad():
                if core:
                    acts = self.core(data)
                else:
                    acts = self.forward(data)
                acts = acts.cpu().numpy()
                acts = acts.reshape(acts.shape[0], -1)
            if k==0:
                activations = np.zeros((nimg, *acts.shape[1:]), 'float32')
            activations[inds] = acts
        return activations
    

def build_model(NN, input_Ly=66, input_Lx=130, n_layers=2, n_conv=16, n_conv_mid=320, use_sensorium_normalization=True, \
    pool=True, depth_separable=True, avgpool=False, use_bn=True, kernel_size = [25, 9], \
    spatial_nconv=False, lessconv=None, minimodel_activation='relu', Wc_coef=0.01, activation='elu'):
    if use_sensorium_normalization: loss_function = 'poisson'
    else: loss_function = 'mse'

    dense = False
    if pool: stride = 1
    else: stride = 2 
    in_channels = [1, n_conv]

    for n in range(1, n_layers):
        in_channels.append(n_conv_mid)
        if n > 1:
            kernel_size.append(5)
    if NN==1:
        if lessconv is not None: true_lessconv = lessconv
        else: true_lessconv = True
        core = Core(in_channels, kernel_size, stride, 
                                dense=dense, depth_separable=depth_separable, pool=pool, activation=minimodel_activation, avgpool=avgpool, batchnorm=use_bn, lessconv=true_lessconv)#[::2])
    else:
        core = Core(in_channels, kernel_size, stride, 
                                dense=dense, depth_separable=depth_separable, pool=pool, spatial_nconv=spatial_nconv, activation=activation, batchnorm=use_bn, avgpool=avgpool)
    in_shape = (in_channels[-1], input_Ly//2, input_Lx//2)
    print('input shape of readout: ', in_shape)

    readout = Readout(in_shape, NN, rank=1,
                                    yx_separable=True, bias_init=None, poisson=use_sensorium_normalization, Wc_coef=Wc_coef) #-spks_mean[ineur])

    model = Encoder(core, readout, loss_fun=loss_function)
    return model, in_channels


def create_model_name(mouse_name, expdate, n_layers, in_channels, clamp=True, use_sensorium_normalization=True, depth_separable=True, \
                      ineuron=-1, seed=1, suffix=False, hs_readout=0.0):
    if mouse_name == 'L1_A1': mouse_name = 'l1a1'
    elif mouse_name == 'L1_A5': mouse_name = 'l1a5'
    model_save_name = f'{mouse_name}_{expdate}_{n_layers}layer'
    for nc in in_channels[1:]:
        model_save_name += f'_{nc}'
    if clamp:
        model_save_name += '_clamp'
    if use_sensorium_normalization:
        model_save_name += '_norm'
    if depth_separable:
        model_save_name += '_depthsep'
    model_save_name += '_pool'
    if ineuron >= 0: # for minimodel
        model_save_name += f'_nn{ineuron}'
        if hs_readout > 0:
            model_save_name += f'_hs{hs_readout:.0e}'
    if suffix:
        model_save_name += f'_{suffix}'
    if seed != 1:
        model_save_name += f'_seed{seed}'
    model_path = model_save_name + '.pt'
    print('model name: ', model_path)
    return model_path

