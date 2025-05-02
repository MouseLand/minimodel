import os
import sys
import torch
import numpy as np

from minimodel import data, model_builder, metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mouse_id', type=int, default=1, help='mouse id')
parser.add_argument('--ineuron', type=int, default=1, help='index of neuron')
parser.set_defaults(normalize=False)
args = parser.parse_args()

mouse_id = args.mouse_id

# load txt16 data
fname = 'text16_%s_%s.npz'%(data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
dat = np.load(os.path.join('../data', fname), allow_pickle=True)
txt16_spks_test = dat['ss_all']
nstim, nrep, nneuron = txt16_spks_test.shape
txt16_istim_test = dat['ss_istim'].astype(int)
txt16_istim_test = np.repeat(txt16_istim_test[:, np.newaxis], nrep, axis=1).flatten()
txt16_spks_test = txt16_spks_test.reshape(-1, nneuron)
txt16_labels_test = dat['ss_labels']
txt16_labels_test = np.repeat(txt16_labels_test[:, np.newaxis], nrep, axis=1).flatten()

print('txt16_spks_test shape:', txt16_spks_test.shape)
print('txt16_labels_test shape:', txt16_labels_test.shape)


txt16_spks_train = dat['sp'].T
txt16_istim_train = dat['istim'].astype(int)
txt16_labels_train = dat['labels']

print('txt16_spks_train shape:', txt16_spks_train.shape)
print('txt16_labels_train shape:', txt16_labels_train.shape)

txt16_spks = np.vstack((txt16_spks_train, txt16_spks_test))
txt16_labels = np.hstack((txt16_labels_train, txt16_labels_test))
txt16_istim = np.hstack((txt16_istim_train, txt16_istim_test))

print('txt16_spks shape:', txt16_spks.shape)
print('txt16_labels shape:', txt16_labels.shape)

# load valid neurons
dat = np.load(f'../figures/save_results/outputs/fullmodel_{data.db[mouse_id]["mname"]}_results.npz', allow_pickle=True)
valid_idxes = dat['valid_idxes']

dat = np.load(f'../figures/save_results/outputs/minimodel_{data.db[mouse_id]["mname"]}_result.npz', allow_pickle=True)
fev_all = dat['fev_all']
feve_all = dat['feve_all']

print('valid_idxes shape:', valid_idxes.shape)
print('fev_all shape:', fev_all.shape)
print('feve_all shape:', feve_all.shape)

# only keep neurons with FEVE>0.7
valid_ineurons = valid_idxes[feve_all>0.7]
print('valid_ineurons shape:', valid_ineurons.shape)

device = torch.device('cpu')
depth_separable = True
pool = True
clamp = True
use_30k = False # use all data recorded (>30k) or only 30k, performance will decrease if use only 30k.
server_path = '../'


op_list = ['conv2_1x1', 'conv2_spatial', 'conv2_relu', 'Wxy', 'elu']

# load txt16 images
data_path = '../data'
if mouse_id == 5:
    xrange_max = 176
else:
    xrange_max = 130
img = data.load_images(data_path, file=os.path.join(data_path, 'nat60k_text16.mat'), normalize=False, xrange=[xrange_max-130,xrange_max])
# img = data.load_images_mat(img_root, file='nat60k_text16.mat', downsample=1, normalize=True, crop=False, origin=True)[0]
print('img: ', img.shape, img.min(), img.max(), img.dtype)

txt16_img = img[txt16_istim]
print(txt16_img.shape, txt16_img.max(), txt16_img.min())

# zscore txt16_imgs
img_mean = txt16_img.mean()
img_std = txt16_img.std()
txt16_img_zscore = (txt16_img - img_mean) / img_std
print(txt16_img_zscore.shape, txt16_img_zscore.max(), txt16_img_zscore.min())
txt16_img_zscore = torch.from_numpy(txt16_img_zscore).to(device).unsqueeze(1)
print(txt16_img_zscore.shape)


# build model
seed = 1
nconv1 = 16
nconv2 = 64
nlayers = 2
wc_coef = 0.2
hs_readout = 0.03

ineur = [valid_ineurons[args.ineuron]]

save_path = f'./catvar/{data.mouse_names[mouse_id]}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_file_name = f'{data.mouse_names[mouse_id]}_minimodel_{nconv1}_{nconv2}_pairwise_catvar_neuron{ineur[0]}_result.npz'
save_path = os.path.join(save_path, save_file_name)

if not os.path.exists(save_path):
    suffix  = ''
    if mouse_id == 5: suffix += 'xrange_176'
    model, in_channels = model_builder.build_model(NN=1, n_layers=nlayers, n_conv=nconv1, n_conv_mid=nconv2, pool=pool, depth_separable=depth_separable, Wc_coef=wc_coef)
    model_name = model_builder.create_model_name(data.mouse_names[mouse_id], data.exp_date[mouse_id], ineuron=ineur[0], n_layers=nlayers, in_channels=in_channels, clamp=clamp, seed=seed,hs_readout=hs_readout, suffix=suffix)
    weight_path = os.path.join(server_path, 'weights', 'minimodel', data.mouse_names[mouse_id])
    model_path = os.path.join(weight_path, model_name)
    print('model path: ', model_path)
    model.load_state_dict(torch.load(model_path))
    print('loaded model', model_path)
    model = model.to(device)

    data_dict = {}

    with torch.no_grad():
        Wc = model.readout.Wc.cpu().detach().numpy().squeeze()
        valid_channels = np.where(np.abs(Wc)>0.01)[0]

        # get conv2 features and catvar
        model.eval()
        conv1_pool_fvs = model.core.features.layer0(txt16_img_zscore)
        print('after conv1_pool: ', conv1_pool_fvs.shape, conv1_pool_fvs.max(), conv1_pool_fvs.min())

        conv2_1x1_fvs = model.core.features.layer1.ds_conv.in_depth_conv(conv1_pool_fvs)
        print('after conv2_1x1: ', conv2_1x1_fvs.shape, conv2_1x1_fvs.max(), conv2_1x1_fvs.min())

        del conv1_pool_fvs

        conv2_spatial_fvs = model.core.features.layer1.ds_conv.spatial_conv(conv2_1x1_fvs)
        print('after conv2_spatial: ', conv2_spatial_fvs.shape, conv2_spatial_fvs.max(), conv2_spatial_fvs.min())

        conv2_bn_fvs = model.core.features.layer1.norm(conv2_spatial_fvs)
        print('after conv2_bn: ', conv2_bn_fvs.shape, conv2_bn_fvs.max(), conv2_bn_fvs.min())

        conv2_relu_fvs = model.core.features.layer1.activation(conv2_bn_fvs)
        print('after conv2_relu: ', conv2_relu_fvs.shape, conv2_relu_fvs.max(), conv2_relu_fvs.min())

        del conv2_bn_fvs    

        wxy_fvs = torch.einsum('iry, irx, ncyx -> ncr', model.readout.Wy, model.readout.Wx, conv2_relu_fvs)
        print('after wxy: ', wxy_fvs.shape, wxy_fvs.max(), wxy_fvs.min())
        data_dict['channel_activities'] = wxy_fvs.cpu().detach().numpy().squeeze()

        wc_fvs = torch.einsum('nrc, ncr -> nr', model.readout.Wc, wxy_fvs)
        print('after wc: ', wc_fvs.shape, wc_fvs.max(), wc_fvs.min())

        elu_fvs = model.readout.activation(wc_fvs+model.readout.bias)
        print('after elu: ', elu_fvs.shape, elu_fvs.max(), elu_fvs.min())


    model_fvs_all = [conv2_1x1_fvs.cpu().detach().numpy()[:, valid_channels], conv2_spatial_fvs.cpu().detach().numpy()[:, valid_channels], conv2_relu_fvs.cpu().detach().numpy()[:, valid_channels], \
                    wxy_fvs.cpu().detach().numpy()[:, valid_channels], elu_fvs.cpu().detach().numpy()]

    del conv2_1x1_fvs
    del conv2_spatial_fvs
    del conv2_relu_fvs
    del wxy_fvs
    del wc_fvs
    del elu_fvs

    catvar_all = np.zeros(len(op_list))
    for i in range(len(op_list)):
        fv = model_fvs_all[i].reshape(model_fvs_all[i].shape[0], -1) # (nstim, nfeatures)
        if len(fv.shape) == 1:
            fv = fv[:, np.newaxis]
        cat_var = metrics.category_variance_pairwise(fv.T, txt16_labels)
        catvar_all[i] = np.nanmean(cat_var)

    data_dict['catvar'] = catvar_all
    data_dict['op_names'] = op_list
    with torch.no_grad():
        pred = model(txt16_img_zscore)
    data_dict['pred'] = pred.cpu().detach().numpy().squeeze()

    np.savez(save_path, **data_dict)
