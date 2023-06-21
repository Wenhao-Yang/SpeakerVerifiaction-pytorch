#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: output_extract.py
@Time: 2020/3/21 5:57 PM
@Overview:
"""
from __future__ import print_function

import argparse
import shap
import json
import os
import pdb
import pickle
import random
import time
from collections import OrderedDict
from Process_Data.Datasets.LmdbDataset import Hdf5DelectDataset
import Process_Data.constants as c
import h5py

import numpy as np
import torch
import torch._utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from kaldi_io import read_mat
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from Define_Model.SoftmaxLoss import AngleLinear, AdditiveMarginLinear
# from Define_Model.model import PairwiseDistance
from Process_Data.Datasets.KaldiDataset import ScriptTrainDataset, \
    ScriptTestDataset, ScriptValidDataset
from Process_Data.audio_processing import ConcateOrgInput, mvnormal, ConcateVarInput, read_WaveInt
from TrainAndTest.common_func import create_model, load_model_args, args_model, args_parse

# Version conflict
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
import warnings

warnings.filterwarnings("ignore")

# Training settings
args = args_parse('PyTorch Speaker Recognition: Gradient')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.multiprocessing.set_sharing_strategy('file_system')

if args.cuda:
    cudnn.benchmark = True

# Define visulaize SummaryWriter instance
kwargs = {'num_workers': args.nj, 'pin_memory': False} if args.cuda else {}
l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6) if args.cos_sim else nn.PairwiseDistance(2)

if args.test_input == 'var':
    transform = transforms.Compose([
        ConcateOrgInput(remove_vad=args.remove_vad),
    ])
elif args.test_input == 'fix':
    transform = transforms.Compose([
        ConcateVarInput(remove_vad=args.remove_vad),
    ])

# file_loader = read_mat
# train_dir = ScriptTrainDataset(dir=args.train_dir, samples_per_speaker=args.input_per_spks,
#                                loader=file_loader, transform=transform, return_uid=True)

train_dir = Hdf5DelectDataset(select_dir=args.select_input_dir, 
                              transform=transform)

cam_layers = args.cam_layers

out_feature_grads = []
in_feature_grads = []
in_layer_feat = []
out_layer_feat = []
in_layer_grad = []
out_layer_grad = []

bias_layers = []
biases = []
handlers = []

def extract_layer_bias(module):
    # extract bias of each layer
    if isinstance(module, torch.nn.BatchNorm2d):
        #         print('bn2')
        b = - (module.running_mean * module.weight
               / torch.sqrt(module.running_var + module.eps)) + module.bias
        return b.data
    elif module.bias is None:
        return None
    else:
        return module.bias.data

def _extract_layer_grads(module, in_grad, out_grad):
    # function to collect the gradient outputs
    # from each layer
#     print(module._get_name())
#     print('Input_grad shape:', in_grad[0].shape)
#     print('Output_grad shape:', out_grad[0].shape)
    global in_feature_grads
    global out_feature_grads
    if not module.bias is None:
        in_feature_grads.append(in_grad[0].detach())
        out_feature_grads.append(out_grad[0].detach())

def _extract_layer_feat(module, input, output):
    # function to collect the gradient outputs from each layer
    #     print(module._get_name())
    #     print('Input_grad shape:', in_grad[0].shape)
    #     print('Output_grad shape:', out_grad[0].shape)
    #     if not module.bias is None:
    global in_layer_feat
    global in_layer_feat
    in_layer_feat.append(input[0].detach())
    out_layer_feat.append(output[0].detach())

def _extract_layer_grad(module, in_grad, out_grad):
    # function to collect the gradient outputs from each layer
    #     print(module._get_name())
    #     print('Input_grad shape:', in_grad[0].shape)
    #     print('Output_grad shape:', out_grad[0].shape)
    global in_layer_grad
    global out_layer_grad
    in_layer_grad.append(in_grad[0].detach())
    out_layer_grad.append(out_grad[0].detach())


def calculate_outputs_and_gradients(inputs, model, target_label_idx,
                                    internal_batch=5):
    """sumary_line
    
    Keyword arguments:
    internal_batch -- internal batch_size has little effect on saliency mapplings.
    Return: return_description
    """
    
    # do the pre-processing
    
    if internal_batch != 1:
        inputs = torch.cat(inputs)
        real_inputs = []
        for i in range(len(inputs)):
            if i % internal_batch == 0:
                real_inputs.append(inputs[i:(i+internal_batch)])
        
        inputs = real_inputs
    
    gradients = []
    for s in inputs:
        s = Variable(s.cuda(), requires_grad=True)

        output, _ = model(s)
        output = F.softmax(output, dim=1)

        if target_label_idx is None:
            target_label_idx = torch.argmax(output, 1).item()

        index = torch.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        if s.is_cuda:
            index = index.cuda()

        output = output.gather(1, index)
        
        if internal_batch != 1:
            output = output.sum()

        # clear grad
        model.zero_grad()
        output.backward()

        gradient = s.grad.detach()#.cpu()
        gradients.append(gradient)

    gradients = torch.cat(gradients)

    return gradients, target_label_idx


def padding_baselines(data, baselines):
    this_baselinses = []
    for the_data in baselines:
        while the_data.shape[1] < data.shape[-2]:
            the_data = np.concatenate([the_data, the_data], axis=1)
        
        the_data = torch.tensor(the_data[:, :data.shape[-2]]).float().unsqueeze(0)
        this_baselinses.append(the_data)
        
    return torch.cat(this_baselinses, dim=0)


def train_extract(train_loader, model, file_dir, set_name, 
                  baselines=None, save_per_num=2500):
    # switch to evaluate mode
    model.eval()

    # input_grads = []
    # inputs_uids = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    global out_feature_grads
    global in_feature_grads
    global in_layer_feat
    global out_layer_feat
    global in_layer_grad
    global out_layer_grad
    softmax = nn.Softmax(dim=1)

    zeros = nn.ZeroPad2d(1)

    correct = .0
    total = .0
    max_inputs = 0

    # data_file = file_dir + '/data.h5py'
    grad_file = file_dir + '/grad.h5py'

    with  h5py.File(grad_file, 'w') as gf:
        for batch_idx, (data, label, uid) in pbar:
            # data: torch.Size([1, 1, 813, 80])
            label = torch.LongTensor(label)
            max_inputs = max(max_inputs, data.shape[-2])

            data_lenght = data.shape[-2]
            # max_inputs = max(max_inputs, data.shape[-2])
            max_lenght =  4 * c.NUM_FRAMES_SPECT

            if data.shape[2] >= max_lenght:
                num_segs = int(np.ceil(data.shape[2] / max_lenght))
                if data.shape[2] < (num_segs*max_lenght):
                    data = torch.cat([data, data], dim=-2)
                    data = data[:,:,:int(num_segs*max_lenght)]
                data = data.chunk(num_segs, dim=2)
                # data = torch.cat(x, dim=0)
            else:
                data = [data]

            torch.cuda.empty_cache()

            grads = []
            for the_data in data:
                
                out_feature_grads = []
                in_feature_grads = []
                in_layer_feat = []
                out_layer_feat = []
                in_layer_grad = []
                out_layer_grad = []

                the_data = Variable(the_data.cuda(), requires_grad=True)
                ups = torch.nn.UpsamplingBilinear2d(size=the_data.shape[-2:])
                
                if args.cam in ['gradient', 'grad_cam', 'grad_cam_pp', 'fullgrad', 'acc_grad', 'layer_cam', 'acc_input']:
                    logit, _ = model(the_data)
                    
                    if args.softmax:
                        logit = softmax(logit)

                    logit[0][label.long()].backward()
                    
                    total += 1
                    predicted = torch.max(logit, dim=1)[1]
                    correct += (predicted.cpu() == label.cpu()).sum().item()
                    
                    if args.cam == 'gradient':
                        grad = the_data.grad  # .cpu().numpy().squeeze().astype(np.float32)
                        
                    elif args.cam == 'grad_cam':
                        grad = torch.zeros_like(the_data)
                        L = len(cam_layers)
                        # assert len(out_layer_grad) == L, print(len(out_layer_grad))

                        last_grad = out_layer_grad[0]
                        feat = out_layer_feat[-1]

                        weight = last_grad.mean(dim=(2, 3), keepdim=True)
                        weight /= weight.sum()

                        T = (feat * weight).sum(dim=1, keepdim=True).clamp_min(0)
                        if args.zero_padding and T.shape[-1] < grad.shape[-1]:
                            T = zeros(T)

                        grad += ups(T) #.abs()

                    elif args.cam == 'grad_cam_pp':
                        # grad cam ++ last
                        last_grad = out_layer_grad[0]
                        last_feat = out_layer_feat[-1]
                        
                        first_derivative = logit[0][label.long()].exp(
                        ) * last_grad
                        alpha = last_grad.pow(2) / (
                            2 * last_grad.pow(2) + (last_grad.pow(3) * last_feat).sum(dim=(2, 3), keepdim=True))
                        weight = alpha * (first_derivative.clamp_min_(0))
                        # weight = alpha * (first_derivative.abs())

                        weight = weight.mean(dim=(2, 3), keepdim=True)
                        weight /= weight.sum()

                        grad = (last_feat * weight).sum(dim=1, keepdim=True).clamp_min(0)

                        if args.zero_padding and grad.shape[-1] < the_data.shape[-1]:
                            grad = zeros(grad)

                        grad = ups(grad)

                    elif args.cam == 'fullgrad':
                        # full grad
                        with torch.no_grad():
                            input_gradient = (the_data.grad * the_data)
                            
                            grad = input_gradient.abs()
                            grad = (grad - grad.min()) / (grad.max()- grad.min() + 1e-8)

                            L = len(bias_layers)
                            for j, l in enumerate(bias_layers):
                                bias = biases[L - j - 1]
                                if len(bias.shape) == 1:
                                    bias = bias.reshape(1, -1, 1, 1)
                                grads_shape = out_feature_grads[j].shape

                                if len(grads_shape) == 3:
                                    # pdb.set_trace()
                                    if bias.shape[1] == grads_shape[-1]:
                                        if model.avgpool != None and bias.shape[1] % model.avgpool.output_size[1] == 0:
                                            bias = bias.reshape(
                                                1, -1, 1, model.avgpool.output_size[1])
                                            
                                            out_feature_grads[j] = out_feature_grads[j].reshape(1, -1, grads_shape[1],
                                                                                                model.avgpool.output_size[
                                                                                                    1])
                                        else:
                                            out_feature_grads[j] = out_feature_grads[j].reshape(1, bias.shape[1],
                                                                                                grads_shape[1], -1)

                                # bias = bias.expand_as(out_feature_grads[j])
                                bias_grad = (
                                    out_feature_grads[j] * bias).mean(dim=1, keepdim=True).abs()
                                bias_grad -= bias_grad.min()
                                bias_grad /= bias_grad.max() + 1e-8

                                if args.zero_padding and bias_grad.shape[-1] < input_gradient.shape[-1]:
                                    bias_grad = zeros(bias_grad)
                                
                                grad += ups(bias_grad)

                            # grad /= grad.max()

                    elif args.cam == 'acc_grad':
                        # pdb.set_trace()
                        input_gradient = (the_data.grad * the_data)
                        acc_grad = input_gradient.clone().clamp_min(0)  # .cpu()
                        acc_grad /= acc_grad.max()

                        for i in range(len(out_layer_grad)):
                            # this_grad = out_layer_grad[i].clone()  # .cpu()
                            # this_feat = out_layer_feat[-1 - i].clone()  # .cpu()
                            # print(this_grad.shape, this_feat.shape)
                            # try:
                            this_grad = ups(
                                out_layer_grad[i] * out_layer_feat[-1 - i]).mean(dim=1, keepdim=True)
                            # except Exception as e:
                            #     print(this_grad.shape, this_feat.shape)

                            this_grad = this_grad.clamp_min(0)
                            this_grad /= this_grad.max()

                            acc_grad += this_grad
                        grad = acc_grad / acc_grad.sum()

                    elif args.cam == 'layer_cam':
                        # pdb.set_trace()
                        input_gradient = (the_data.grad.clamp_min(0) * the_data)
                        grad = input_gradient.clone()
                        # acc_grad /= acc_grad.max()

                        for i in range(len(out_layer_grad)):
                            this_grad = ups(out_layer_grad[i].clamp_min(
                                0) * out_layer_feat[-1 - i]).mean(dim=1, keepdim=True)

                            if args.zero_padding and this_grad.shape[-1] < input_gradient.shape[-1]:
                                this_grad = zeros(this_grad)

                            this_grad = ups(this_grad).sum(dim=1, keepdim=True).clamp_min(0)
                            grad += this_grad #/ this_grad.max()

                    elif args.cam == 'acc_input':
                        # pdb.set_trace()
                        grad = torch.zeros_like(the_data)
                        # acc_grad /= acc_grad.max()

                        for i in range(len(out_layer_feat)):
                            # this_grad = out_layer_grad[i].clone().cpu()
                            this_feat = out_layer_feat[-1 - i].clone()
                            if len(this_feat.shape) == 3:
                                this_feat = this_feat.unsqueeze(0)
                            # try:
                            this_grad = ups(this_feat).mean(
                                dim=1, keepdim=True)
                            # except Exception as e:
                            #     print(this_feat.shape)

                            this_grad = this_grad.clamp_min(0)
                            this_grad /= this_grad.max() + 1e-6

                            if np.isnan(this_grad.detach().cpu().numpy()).any():
                                pdb.set_trace()

                            grad += this_grad if args.layer_weight else (len(out_layer_feat) - i) / len(
                                out_layer_feat) * this_grad

                        grad = grad / grad.sum()

                elif args.cam in ['integrad', 'integrad2']:

                    if baselines is None:
                        baseline = 0. * the_data
                    else:
                        baseline = padding_baselines(the_data, baselines)
                        baseline = baseline.mean(dim=0, keepdim=True)

                    baseline = baseline.cuda()
                    scaled_inputs = [baseline + (float(i) / args.steps) * (the_data - baseline) for i in
                                    range(1, args.steps + 1)]

                    the_grads, _ = calculate_outputs_and_gradients(
                        scaled_inputs, model, label)
                    
                    if args.cam == 'integrad':
                        the_grads = (the_grads[:-1] + the_grads[1:]) / 2.0
                    else:
                        the_grads = the_grads[:-1]
                                
                    avg_grads = the_grads.mean(dim=0)
                    grad = (the_data - baseline) * avg_grads  # shape: <grad.shape>
                
                elif args.cam in ['exptgrad']:
                    baseline = padding_baselines(the_data, baselines)
                    explainer = shap.GradientExplainer(model, baseline.cuda(), local_smoothing=0.5)
                    shap_values, _ = explainer.shap_values(data.cuda(),
                                                           ranked_outputs=1,
                                                           nsamples=200)
                    
                    grad = shap_values[0]
            
                if grad.shape != the_data.shape:
                    print(grad.shape, the_data.shape)
                    pdb.set_trace()

                grads.append(grad.cpu().detach().numpy().squeeze().astype(np.float32))

            grad = np.concatenate(grads, axis=0)[:data_lenght]

            out_feature_grads = []
            in_feature_grads = []
            in_layer_feat = []
            out_layer_feat = []
            in_layer_grad = []
            out_layer_grad = []
            # df.create_dataset(uid[0], data=data)
            gf.create_dataset(uid[0], data=grad)
            # inputs_uids.append([uid[0], int(label.numpy()[0])])

            model.zero_grad()
            if batch_idx % args.log_interval == 0:
                pbar.set_description('Saving {} : [{:8d}/{:8d} ({:3.0f}%)] '.format(
                    uid,
                    batch_idx + 1,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
            # if batch_idx % 50 == 1:

    try:
        print('Predict Accuracy: {:.2f}%...'.format(correct/total*100))
    except Exception as e:
        pass

    print('Saving pairs in %s.\n' % file_dir)
    torch.cuda.empty_cache()


def test_extract(test_loader, model, file_dir, set_name, save_per_num=1500):
    # switch to evaluate mode
    model.eval()

    input_grads = []
    inputs_uids = []
    pbar = tqdm(enumerate(test_loader))

    # for batch_idx, (data_a, data_b, label) in pbar:
    for batch_idx, (data_a, data_b, label, uid_a, uid_b) in pbar:

        # pdb.set_trace()
        data_a = Variable(data_a.cuda(), requires_grad=True)
        data_b = Variable(data_b.cuda(), requires_grad=True)

        _, feat_a = model(data_a)
        _, feat_b = model(data_b)

        cos_sim = l2_dist(feat_a, feat_b)
        cos_sim[0].backward()

        grad_a = data_a.grad.cpu().numpy().squeeze().astype(np.float32)
        grad_b = data_b.grad.cpu().numpy().squeeze().astype(np.float32)
        data_a = data_a.data.cpu().numpy().squeeze().astype(np.float32)
        data_b = data_b.data.cpu().numpy().squeeze().astype(np.float32)

        if args.revert:
            grad_a = grad_a.transpose()
            data_a = data_a.transpose()

            grad_b = grad_b.transpose()
            data_b = data_b.transpose()

        input_grads.append((label, grad_a, grad_b, data_a, data_b))
        inputs_uids.append([uid_a, uid_b])

        model.zero_grad()

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Saving pair [{:8d}/{:8d} ({:3.0f}%)] '.format(
                batch_idx + 1,
                len(test_loader),
                100. * batch_idx / len(test_loader)))

        if (batch_idx + 1) % save_per_num == 0 or (batch_idx + 1) == len(test_loader.dataset):
            num = batch_idx // save_per_num if batch_idx + 1 % save_per_num == 0 else batch_idx // save_per_num + 1
            # checkpoint_dir / extract / < dataset > / < set >.*.bin

            filename = file_dir + '/%s.%d.bin' % (set_name, num)
            # print('Saving pairs in %s.' % filename)

            with open(filename, 'wb') as f:
                pickle.dump(input_grads, f)

            with open(file_dir + '/inputs.%s.%d.json' % (set_name, num), 'w') as f:
                json.dump(inputs_uids, f)

            input_grads = []
            inputs_uids = []
    print('Saving pairs into %s.\n' % file_dir)
    torch.cuda.empty_cache()


def main():
    if args.verbose > 1:
        print('\nNumber of Speakers: {}.'.format(train_dir.num_spks))
        # print the experiment configuration
        print('Current time is \33[91m{}\33[0m.'.format(str(time.asctime())))
    
        options = vars(args)
        options_keys = list(options.keys())
        options_keys.sort()
        options_str = ''
        for k in options_keys:
            options_str += '\'{}\': \'{}\', '.format(k, options[k])
        print('Parsed options: \n {}'.format(options_str))

    # instantiate model and initialize weights
    if args.check_yaml != None and os.path.exists(args.check_yaml):
        if args.verbose > 0:
            print('\nLoading model check yaml from: \n\t{}'.format(args.check_yaml.lstrip('Data/checkpoint/')))
        model_kwargs = load_model_args(args.check_yaml)
    else:
        print('Error in finding check yaml file:\n{}'.format(args.check_yaml))
        exit(0)
        # model_kwargs = args_model(args, train_dir)

    if 'embedding_model' in model_kwargs:
        model = model_kwargs['embedding_model']
        if 'classifier' in model_kwargs:
            model.classifier = model_kwargs['classifier']
    else:
        if args.verbose > 0: 
            keys = list(model_kwargs.keys())
            keys.sort()
            model_options = ["\'%s\': \'%s\'" % (str(k), str(model_kwargs[k])) for k in keys]
            print('Model options: \n{ %s }' % (', '.join(model_options)))
            print('Testing with %s distance, ' % ('cos' if args.cos_sim else 'l2'))

        model = create_model(args.model, **model_kwargs)

    train_loader = DataLoader(train_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    # veri_loader = DataLoader(veri_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    # valid_loader = DataLoader(valid_part, batch_size=args.batch_size, shuffle=False, **kwargs)
    # test_loader = DataLoader(test_dir, batch_size=args.batch_size, shuffle=False, **kwargs)
    # sitw_test_loader = DataLoader(sitw_test_part, batch_size=args.batch_size, shuffle=False, **kwargs)
    # sitw_dev_loader = DataLoader(sitw_dev_part, batch_size=args.batch_size, shuffle=False, **kwargs)

    resume_path = args.check_path + '/checkpoint_{}.pth'
    if args.verbose > 1:
        print('=> Saving output in {}\n'.format(args.extract_path))
    epochs = np.arange(args.start_epochs, args.epochs + 1)

    for ep in epochs:
        # Load model from Checkpoint file
        if os.path.isfile(resume_path.format(ep)):
            if args.verbose > 1:
                print('=> loading checkpoint {}'.format(resume_path.format(ep)))
            checkpoint = torch.load(resume_path.format(ep))
            checkpoint_state_dict = checkpoint['state_dict']
            if isinstance(checkpoint_state_dict, tuple):
                checkpoint_state_dict = checkpoint_state_dict[0]

            # epoch = checkpoint['epoch']
            # if e == 0:
            #     filtered = checkpoint.state_dict()
            # else:
            filtered = {k: v for k, v in checkpoint_state_dict.items() if 'num_batches_tracked' not in k}
            if list(filtered.keys())[0].startswith('module'):
                new_state_dict = OrderedDict()
                for k, v in filtered.items():
                    name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，去掉module.
                    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

                model.load_state_dict(new_state_dict)
            else:
                model_dict = model.state_dict()
                model_dict.update(filtered)
                model.load_state_dict(model_dict)
        else:
            print('=> no checkpoint found at %s' % resume_path.format(ep))
            continue

        if args.feat_format == 'wav':
            trans = model.input_mask[0]
            model.input_mask.__delitem__(0)
            transform.transforms.append(trans)
            
        baselines = None
        if os.path.exists(args.baseline_file):
            baselines = [] 
            with open(args.baseline_file, 'r') as f:
                for l in f.readlines():
                    _, upath = l.split()
                    the_data = read_WaveInt(upath)
                    the_data = trans(torch.tensor(the_data).reshape(1, 1, -1).float())
                    baselines.append(the_data)
                    
            if args.verbose > 1:
                print('Baselines shape: ', len(baselines))

        model.cuda()

        # cam_layers = ['conv1', 'layer1.2.conv2']
        global bias_layers
        global biases
        global handlers

        valid_layers = []
        if args.cam in ['grad_cam', 'grad_cam_pp', 'fullgrad', 'acc_grad', 'acc_input', 'layer_cam']:
            for name, m in model.named_modules():
                try:
                    if name in cam_layers:
                        valid_layers.append(name)
                        handlers.append(
                            m.register_forward_hook(_extract_layer_feat))
                        handlers.append(
                            m.register_backward_hook(_extract_layer_grad))

                    if not ('fc' in name or 'classifier' in name or 'CBAM' in name):
                        b = extract_layer_bias(m)
                        if (b is not None):
                            biases.append(b.detach())
                            bias_layers.append(name)
                            m.register_backward_hook(_extract_layer_grads)

                except Exception as e:
                    continue

        if args.verbose > 0 and args.cam == 'fullgrad':
            print("The number of layers with biases: {}".format(len(biases)))
        if args.verbose > 0 :
            print("Valid layers for {}: {}".format(args.cam, " ".join(valid_layers)))

        print('')

        file_dir = args.extract_path # + '/epoch_%d' % ep
        if args.cam == 'acc_input' and args.layer_weight:
            file_dir += '_layer_weight'

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        train_extract(train_loader, model, file_dir,
                          '%s_train' % args.train_set_name, baselines=baselines)


if __name__ == '__main__':
    main()
