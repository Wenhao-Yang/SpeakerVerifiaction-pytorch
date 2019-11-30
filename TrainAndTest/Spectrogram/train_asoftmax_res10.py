#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: test_accuracy.py
@Time: 19-8-6 下午1:29
@Overview: Train the resnet 10 with asoftmax. The acoustic feature will be spectrogram, which has dimension of 300*257 for each frame.
"""
from __future__ import print_function
import argparse
import pathlib
import pdb
import time

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os

import numpy as np
from tqdm import tqdm
from Define_Model.model import ResSpeakerModel
from Define_Model.model import SuperficialResNet
from Process_Data.VoxcelebTestset import VoxcelebTestset
from eval_metrics import evaluate_kaldi_eer

from logger import Logger

from Process_Data.DeepSpeakerDataset_dynamic import ClassificationDataset
from Process_Data.voxceleb_wav_reader import wav_list_reader

from Define_Model.model import PairwiseDistance
from Process_Data.audio_processing import GenerateSpect, concateinputfromMFB
from Process_Data.audio_processing import toMFB, totensor, truncatedinput, truncatedinputfromMFB,read_MFB,read_audio,mk_MFB
from Process_Data.audio_processing import truncatedinputfromSpectrogram
# Version conflict

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Speaker Recognition')
# Model options
# Todo: change the roor path
parser.add_argument('--dataroot', type=str, default='Data/dataset/voxceleb1/spectrogram/voxceleb1_wav',
                    help='path to dataset')
parser.add_argument('--test-pairs-path', type=str, default='Data/dataset/ver_list.txt',
                    help='path to pairs file')

parser.add_argument('--log-dir', default='data/pytorch_speaker_logs',
                    help='folder to output model checkpoints')

parser.add_argument('--ckp-dir', default='Data/checkpoint/spectrogram',
                    help='folder to output model checkpoints')

# Todo: create and change the dir for checkpoint files
parser.add_argument('--resume',
                    default='Data/checkpoint/spectrogram/soft_res10/checkpoint_45.pth',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=35, metavar='E',
                    help='number of epochs to train (default: 10)')
# Training options
parser.add_argument('--cos-sim', action='store_true', default=True,
                    help='using Cosine similarity')
parser.add_argument('--embedding-size', type=int, default=512, metavar='ES',
                    help='Dimensionality of the embedding')
parser.add_argument('--batch-size', type=int, default=128, metavar='BS',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='BST',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--test-input-per-file', type=int, default=1, metavar='IPFT',
                    help='input sample per file for testing (default: 8)')

# The following is the parameters for triplet loss training.

#parser.add_argument('--n-triplets', type=int, default=1000000, metavar='N',
# parser.add_argument('--n-triplets', type=int, default=100000, metavar='N',
#                     help='how many triplets will generate from the dataset')

# parser.add_argument('--margin', type=float, default=0.1, metavar='MARGIN',
#                     help='the margin value for the triplet loss function (default: 1.0')

# parser.add_argument('--min-softmax-epoch', type=int, default=2, metavar='MINEPOCH',
#                     help='minimum epoch for initial parameter using softmax (default: 2')

parser.add_argument('--margin', type=float, default=3, metavar='MARGIN',
                    help='the margin value for the triplet loss function (default: 1.0')
parser.add_argument('--loss-ratio', type=float, default=2.0, metavar='LOSSRATIO',
                    help='the ratio softmax loss - triplet loss (default: 2.0')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.125)')
parser.add_argument('--lr-decay', default=1e-4, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-4')
parser.add_argument('--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--optimizer', default='adagrad', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=1, metavar='LI',
                    help='how many batches to wait before logging training status')

parser.add_argument('--acoustic-feature', choices=['fbank', 'spectrogram', 'mfcc'], default='spectrogram',
                    help='choose the acoustic features type.')
parser.add_argument('--makemfb', action='store_true', default=False,
                    help='need to make mfb file')
parser.add_argument('--makespec', action='store_true', default=False,
                    help='need to make spectrograms file')

args = parser.parse_args()

# Set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if args.cuda:
    cudnn.benchmark = True
CKP_DIR = args.ckp_dir
LOG_DIR = args.log_dir + '/run-test_{}-lr{}-wd{}-embeddings{}-msceleb-alpha10'\
    .format(args.optimizer, args.lr, args.wd,  args.embedding_size)

# create logger
logger = Logger(LOG_DIR)
# Define visulaize SummaryWriter instance
writer = SummaryWriter('Log/spectrogram/asoft_res10', filename_suffix=str(time.asctime()))

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.cos_sim:
    l2_dist = nn.CosineSimilarity(dim=1, eps=1e-6)
else:
    l2_dist = PairwiseDistance(2)

voxceleb, voxceleb_dev = wav_list_reader(args.dataroot)
if args.makemfb:
    #pbar = tqdm(voxceleb)
    for datum in voxceleb:
        mk_MFB((args.dataroot +'/voxceleb1_wav/' + datum['filename']+'.wav'))
    print("Complete convert")

if args.makespec:
    num_pro = 1.
    for datum in voxceleb:
        # Data/Voxceleb1/
        # /data/voxceleb/voxceleb1_wav/
        GenerateSpect(wav_path='/data/voxceleb/voxceleb1_wav/' + datum['filename']+'.wav',
                      write_path=args.dataroot +'/spectrogram/voxceleb1_wav/' + datum['filename']+'.npy')
        print('\rprocessed {:2f}% {}/{}.'.format(num_pro/len(voxceleb), num_pro, len(voxceleb)), end='\r')
        num_pro += 1
    print('\nComputing Spectrograms success!')
    exit(1)

if args.acoustic_feature=='fbank':
    transform = transforms.Compose([
        concateinputfromMFB(),
        # truncatedinputfromMFB(),
        totensor()
    ])
    transform_T = transforms.Compose([
        # truncatedinputfromMFB(input_per_file=args.test_input_per_file),
        concateinputfromMFB(input_per_file=args.test_input_per_file),
        totensor()
    ])
    file_loader = read_MFB

elif args.acoustic_feature=='spectrogram':
    # Start from spectrogram
    transform = transforms.Compose([
        concateinputfromMFB(),
        #truncatedinputfromSpectrogram(),
        totensor()
    ])
    transform_T = transforms.Compose([
        concateinputfromMFB(input_per_file=args.test_input_per_file),
        #truncatedinputfromSpectrogram(input_per_file=args.test_input_per_file),
        totensor()
    ])
    file_loader = read_MFB

else:
    transform = transforms.Compose([
                        truncatedinput(),
                        toMFB(),
                        totensor(),
                        #tonormal()
                    ])
    file_loader = read_audio

train_dir = ClassificationDataset(voxceleb=voxceleb_dev,
                                  dir=args.dataroot,
                                  loader=file_loader,
                                  transform=transform)

test_dir = VoxcelebTestset(dir=args.dataroot,
                           pairs_path=args.test_pairs_path,
                           loader=file_loader,
                           transform=transform_T)

del voxceleb
del voxceleb_dev


def main():
    # Views the training images and displays the distance on anchor-negative and anchor-positive
    test_display_triplet_distance = False

    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))
    print('\nNumber of Classes:\n{}\n'.format(len(train_dir.classes)))

    # instantiate model and initialize weights
    # model = ResSpeakerModel(embedding_size=args.embedding_size,
    #                         resnet_size=10,
    #                         num_classes=len(train_dir.classes),
    #                         feature_dim=257)
    model = SuperficialResNet(layers=[1, 1, 1, 1],
                              embedding_size=args.embedding_size,
                              n_classes=len(train_dir.classes),
                              m=args.margin)

    if args.cuda:
        model.cuda()

    optimizer = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)

            filtered = {k: v for k, v in checkpoint['state_dict'].items() if 'num_batches_tracked' not in k}

            model.load_state_dict(filtered)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # criterion.load_state_dict(checkpoint['criterion'])

        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    print('start epoch is : ' + str(start))
    # start = 0
    end = start + args.epochs

    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        pdb.set_trace()
        return [data, target]

    train_loader = torch.utils.data.DataLoader(train_dir, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dir, batch_size=args.test_batch_size, shuffle=False, collate_fn=my_collate, **kwargs)

    for epoch in range(start, end):
        # pdb.set_trace()
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch)
        #break

    writer.close()

def train(train_loader, model, optimizer, epoch):
    # switch to evaluate mode
    model.train()
    # labels, distances = [], []
    correct = 0.
    total_datasize = 0.
    total_loss = 0.

    pbar = tqdm(enumerate(train_loader))
    #pdb.set_trace()
    output_softmax = nn.Softmax(dim=1)
    ce = nn.CrossEntropyLoss()

    for batch_idx, (data, label) in pbar:
        if args.cuda:
            data = data.cuda()
        data, label = Variable(data), Variable(label)

        #pdb.set_trace()
        feats = model(data)
        classfier = model.forward_classifier(feats)

        predicted_labels = output_softmax(classfier)
        predicted_one_labels = torch.max(predicted_labels, dim=1)[1]

        true_labels = label.cuda()

        # loss = model.AngularSoftmaxLoss(feats, true_labels.cuda())
        loss = ce(classfier, true_labels)
        # loss = cross_entropy_loss  # + triplet_loss * args.loss_ratio

        minibatch_acc = float((predicted_one_labels.cuda() == true_labels.cuda()).sum().item()) / len(predicted_one_labels)
        correct += float((predicted_one_labels.cuda()==true_labels.cuda()).sum().item())
        total_datasize += len(predicted_one_labels)
        total_loss += loss.item()
        # Visualize loss and acc
        writer.add_scalar('Train_Loss/epoch_%d' % epoch, loss.item(), batch_idx)
        writer.add_scalar('Train_Accuracy/epoch_%d' % epoch, minibatch_acc, batch_idx)
        #pdb.set_trace()

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\tLoss: {:.6f} \tMinibatch Accuracy: {:.6f}%'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                100. * minibatch_acc))

    check_path = pathlib.Path('{}/soft_res10/checkpoint_{}.pth'.format(args.ckp_dir, epoch))
    if not check_path.parent.exists():
        os.makedirs(str(check_path.parent))

    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               # 'criterion': criterion.state_dict()
               str(check_path))

    # torch.save({'epoch': epoch+1,
    #             'state_dict': model.state_dict(),
    #             'optimizer': optimizer.state_dict()},
    #             #'criterion': criterion.state_dict()
    #            '{}/resnet10_asoftmax/checkpoint_{}.pth'.format(CKP_DIR, epoch))


    print('\33[91mFor ASoftmax Train set Accuracy:{:.6f}% \n\33[0m'.format(100 * float(correct) / total_datasize))
    writer.add_scalar('Train_Accuracy_Per_Epoch', correct/total_datasize, epoch)
    writer.add_scalar('Train_Loss_Per_Epoch', total_loss/len(train_loader), epoch)

def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:
        current_sample = data_a.size(0)
        data_a = data_a.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        data_p = data_p.resize_(args.test_input_per_file * current_sample, 1, data_a.size(2), data_a.size(3))
        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()
        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        # compute output
        out_a, out_p = model(data_a), model(data_p)

        dists = l2_dist.forward(out_a, out_p)
        dists = dists.data.cpu().numpy()
        dists = dists.reshape(current_sample, args.test_input_per_file).mean(axis=1)
        distances.append(dists)
        labels.append(label.data.cpu().numpy())

        if batch_idx % args.log_interval == 0:
            pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                100. * batch_idx / len(test_loader)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    # err, accuracy= evaluate_eer(distances,labels)
    eer, eer_threshold, accuracy = evaluate_kaldi_eer(distances, labels, cos=args.cos_sim, re_thre=True)
    writer.add_scalar('Test_Result/eer', eer, epoch)
    writer.add_scalar('Test_Result/threshold', eer_threshold, epoch)
    writer.add_scalar('Test_Result/accuracy', accuracy, epoch)
    # tpr, fpr, accuracy, val, far = evaluate(distances, labels)

    if args.cos_sim:
        print(
            '\33[91mFor cos_distance, Test set ERR is {:.8f} when threshold is {}\tAnd test accuracy could be {:.2f}%.\n\33[0m'.format(
                100. * eer, eer_threshold, 100. * accuracy))
    else:
        print('\33[91mFor l2_distance, Test set ERR: {:.8f}%\tBest ACC:{:.8f} \n\33[0m'.format(100. * eer, accuracy))
    # logger.log_value('Test Accuracy', np.mean(accuracy))


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.99, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=new_lr,
                                  lr_decay=args.lr_decay,
                                  weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()

