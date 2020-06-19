from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc, preproc_tf
from layers.modules import MultiBoxLoss, MultiBoxLoss_tf_source, MultiBoxLoss_tf_source_combine,  KD_loss, search_imprinted_weights, MultiBoxLoss_tf_target, MultiBoxLoss_tf_target_balance
from layers.functions import PriorBox
import time
from data_utlis.low_shot_tramsform import VOC_low_shot_dataset_transform

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# torch.cuda.set_device(4)

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='./weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=50,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=2,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=2e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument(
    '--resume_net', default='weights/task1-source/tradiction/RFB_vgg_VOC_epoches_220.pth', help='resume net for retraining')
    #'--resume_net', default='weights/task3-target1/kd_ls/RFB_vgg_VOC_epoches_395.pth', help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=600,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/task1-target1/test-exp-nopretrain/',
                    help='Location to save checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    #train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    #train_sets = [('2007', 'trainval')]
    #train_sets = [('2007-task3-target1', 'trainval')]
    train_sets = [('2007-CLtask1-target1', 'trainval')]
    #train_sets = [('2007-IL-target_1to10', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    # train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    train_sets = [('2017', 'train')]
    # train_sets = [('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

cfg = COCO_300

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
else:
    print('Unkown version!')

img_dim = (300, 512)[args.size == '512']
rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
p = (0.6, 0.2)[args.version == 'RFB_mobile']
num_classes = (61, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
            elif 'bn' in key:
                m.state_dict()[key][...] = 1
            else:
                xavier(m.state_dict()[key])
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

def weights_zeros_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                m.state_dict()[key][...] = 0
            elif 'bn' in key:
                m.state_dict()[key][...] = 1
            else:
                m.state_dict()[key][...] = 0
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


# load the source
net = build_net('train', img_dim, num_classes)
print(net)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.bin_conf.apply(weights_init)
    net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    print('Loading resume network...')

    #args.resume_net = 'weights/task3-source/RFB_vgg_VOC_epoches_295.pth'
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model_dict = net.state_dict()
    new_state_dict_filter = {k: v for k, v in new_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_state_dict_filter)
    # 3. load the new state dict
    net.load_state_dict(model_dict)

for pars_val in net.parameters():
    pars_val.requires_grad = False

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    # net = torch.nn.DataParallel(net)

if args.cuda:
    net.cuda()
    cudnn.benchmark = True


# load the target1
if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg_target1 import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net

    cfg = COCO_mobile_300
else:
    print('Unkown version!')

# load the source
num_classes_target1 = 6
net_target1 = build_net('train', img_dim, num_classes, num_classes_target1)
print(net_target1)
if args.resume_net == None:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net.base.load_state_dict(base_weights)

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    net.extras.apply(weights_init)
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)
    net.bin_conf.apply(weights_init)
    net.Norm.apply(weights_init)
    if args.version == 'RFB_E_vgg':
        net.reduce.apply(weights_init)
        net.up_reduce.apply(weights_init)

else:
    # load resume network
    print('Loading resume network...')

    #args.resume_net = 'weights/task3-target1/kd_ls/RFB_vgg_VOC_epoches_395.pth'

    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model_dict = net_target1.state_dict()

    # pretrain parameters for bin_conf and loc
    for index_val in range(6):
        new_state_dict['loc_target1.' + str(index_val) + '.weight'] = new_state_dict[
            'loc.' + str(index_val) + '.weight']
        new_state_dict['loc_target1.' + str(index_val) + '.bias'] = new_state_dict[
            'loc.' + str(index_val) + '.bias']
        new_state_dict['bin_conf_target1.' + str(index_val) + '.weight'] = new_state_dict[
            'bin_conf.' + str(index_val) + '.weight']
        new_state_dict['bin_conf_target1.' + str(index_val) + '.bias'] = new_state_dict[
            'bin_conf.' + str(index_val) + '.bias']

    new_state_dict_filter = {k: v for k, v in new_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_state_dict_filter)
    # 3. load the new state dict
    net_target1.load_state_dict(model_dict)
    #net_target1.multi_conf_to_bin_conf_layers_target1.apply(weights_zeros_init)

    net_target1.loc_target1.apply(weights_init)
    #net_target1.conf_target1.apply(weights_init)
    net_target1.bin_conf_target1.apply(weights_init)
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    net_target1.base.load_state_dict(base_weights)


if args.ngpu > 1:
    net_target1 = torch.nn.DataParallel(net_target1, device_ids=list(range(args.ngpu)))
    # net = torch.nn.DataParallel(net)

if args.cuda:
    net_target1.cuda()
    cudnn.benchmark = True

'''
params = net_target1.module.base
for i in range(30):
    for v in params[i].parameters():
        v.requires_grad = False
'''
optimizer = optim.SGD(net_target1.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.RMSprop(net.parameters(), lr=args.lr,alpha = 0.9, eps=1e-08,
#                      momentum=args.momentum, weight_decay=args.weight_decay)

# criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
#criterion = MultiBoxLoss_tf_source(num_classes, 0.5, True, 0, True, 3, 0.5, False)

S_imprinted_weights = search_imprinted_weights(num_classes, 0.75, True, 0, True, 3, 0.5, False)
target_criterion = MultiBoxLoss_tf_target(num_classes_target1, 0.5, True, 0, True, 3, 0.5, False)
#target_criterion = MultiBoxLoss_tf_target_balance(num_classes_target1, 0.5, True, 0, True, 3, 0.5, False)
#criterion = MultiBoxLoss_tf_source_combine(num_classes, 0.5, True, 0, True, 3, 0.5, False)
KD_criterion = KD_loss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

low_shots_num = 1
new_class_num = 6

def train():
    net.train()
    net_target1.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    # low shot tramsform
    dataset = VOC_low_shot_dataset_transform(dataset, low_shots_num, new_class_num)

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (400 * epoch_size, 500 * epoch_size, 600 * epoch_size)
    stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
    stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
    print('Training', args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    lr = args.lr
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                                  shuffle=True, num_workers=args.num_workers,
                                                  collate_fn=detection_collate))
            loc_loss = 0
            conf_loss = 0
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > 200):
                torch.save(net_target1.state_dict(), args.save_folder + args.version + '_' + args.dataset + '_epoches_' +
                           repr(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)

        # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]

        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # forward
        t0 = time.time()
        source_out = net(images)
        target1_out, target1_out1, _ = net_target1(images)

        # backprop
        optimizer.zero_grad()
        #loss_l, loss_c, loss_bin, pos, neg_binary, neg_multi = criterion(target1_out1, priors, targets)

        loss_l, loss_c, loss_bin, pos, neg_binary, neg_multi = target_criterion(target1_out1, priors, targets)
        # loss_l, loss_c, pos, neg = criterion(target1_out1, priors, targets)
        #kd_loss_l, kd_loss_c, kd_loss_bin = KD_criterion(source_out, target1_out,  priors, targets, pos)
        #kd_loss_l, kd_loss_c, kd_loss_bin = KD_criterion(source_out, target1_out, target1_out1, priors, targets, pos, neg_binary, neg_multi)
        kd_loss_l, kd_loss_b, kd_loss_c = KD_criterion(source_out, target1_out, target1_out1, priors, targets, pos, neg_binary, neg_multi)
        loss_target = loss_l + loss_bin + loss_c
        #kd_loss = kd_loss_l + kd_loss_c + kd_loss_bin
        kd_loss = 0 # kd_loss_l + kd_loss_c #+ kd_loss_b

        loss = loss_target + kd_loss
        loss.backward()
        '''
        net_target1.module.classifier_target1.fc.weight.grad *= 10
        for parameter_val in net_target1.module.bin_conf_target1.parameters():
            parameter_val.grad *= 10
        for parameter_val in net_target1.module.loc_target1.parameters():
            parameter_val.grad *= 10
        net_target1.module.scale_target1.grad *= 10
        '''
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()

        net_target1.module.weight_target1_norm()
        if iteration % 10 == 0:
            '''
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f  kd_L: %.4f kd: %.4f||' % (
                      loss_l.item(), loss_c.item(),  kd_loss_l.item(), kd_loss_c.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
            '''
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' +
                  repr(iteration) + ' || L: %.4f C: %.4f  B: %.4f KD_L: %.4f KD_C: %.4f ||' % (
                      loss_l.item(), loss_c.item(), loss_bin.item(), kd_loss_l.item(), kd_loss_c.item()) +
                  'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
    torch.save(net_target1.state_dict(), args.save_folder +
               'Final_' + args.version + '_' + args.dataset + '.pth')



def train_imprint():
    net.train()
    net_target1.train()
    max_epoch = 10
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        dataset = VOCDetection(VOCroot, train_sets, preproc(
            img_dim, rgb_means, p), AnnotationTransform())
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(
            img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    # low shot tramsform
    dataset = VOC_low_shot_dataset_transform(dataset, low_shots_num, new_class_num)
    low_batch = 50
    epoch_size = len(dataset) // low_batch
    start_iter = 0
    max_iter = max_epoch * epoch_size


    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, low_batch, shuffle=True, num_workers=1,
                                                  collate_fn=detection_collate))

        # load train data
        with torch.no_grad():
            images, targets = next(batch_iterator)

            # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda()) for anno in targets]

            else:
                images = Variable(images)
                targets = [Variable(anno) for anno in targets]

            # forward
            out_0, out_1, embedding_target1 = net_target1(images)

            embedding_weights_pos, targets_belongs = S_imprinted_weights(embedding_target1, out_0,  priors, targets)

            if iteration == 0:
                output_stack = embedding_weights_pos
                target_stack = targets_belongs
            else:
                output_stack = torch.cat((output_stack, embedding_weights_pos), 0)
                target_stack = torch.cat((target_stack, targets_belongs), 0)

            print(iteration)

    new_weight = torch.zeros_like(net_target1.module.classifier_target1.fc.weight.data)
    for i in range(1, new_weight.shape[0] + 1):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i - 1] = tmp / tmp.norm(p=2)
    net_target1.module.classifier_target1.fc.weight.data = new_weight




def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    #train_imprint()
    train()
