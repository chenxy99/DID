from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect, PriorBox, Detect_tf, Detect_tf_soft, Detect_tf_soft_cls, Detect_tf_soft_source_cls
from utils.nms_wrapper import nms
from utils.timer import Timer

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/task1-target4/test-exp-nopretrain/Final_RFB_vgg_VOC.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg_target4 import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_netoo
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

cfg = COCO_300

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def vis_picture(im):
    import numpy as np
    import matplotlib.pyplot as plt
    npimg = im.cpu().numpy()
    npimg = np.squeeze(npimg, 0)
    im = np.transpose(npimg, (1, 2, 0))

    im = (im + np.array([104, 117, 123])) / 255
    im = im[:, :, ::-1]
    #im[:,:,0] = 0

    plt.cla()
    plt.imshow(im)
    plt.show()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    # need to change
    num_classes = (6, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        #vis_picture(x)

        _t['im_detect'].tic()
        out, out1, out2, out3, out4, _ = net(x)      # forward pass
        # boxes, scores = detector.forward(out,priors)
        boxes, scores = detector.forward(out1, priors, scale)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = (61, 81)[args.dataset == 'COCO']
    num_classes_target1 = 6
    num_classes_target2 = 6
    num_classes_target3 = 6
    num_classes_target4 = 6
    net = build_net('test', img_dim, num_classes, num_classes_target1, num_classes_target2, num_classes_target3, num_classes_target4)    # initialize detector
    # net.load_state_dict(torch.load(args.trained_model))
    state_dict = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model_dict = net.state_dict()
    new_state_dict_filter = {k: v for k, v in new_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_state_dict_filter)
    # 3. load the new state dict
    #net_target1.load_state_dict(model_dict)

    net.load_state_dict(model_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    if args.dataset == 'VOC':
        testset = VOCDetection(
        #VOCroot, [('2007', 'test')], None, AnnotationTransform())
            #VOCroot, [('2007-task3-source', 'test')], None, AnnotationTransform())
            #VOCroot, [('2007-task3-target1', 'test')], None, AnnotationTransform())
        #VOCroot, [('2017-task1-source-test', 'test')], None, AnnotationTransform())
        VOCroot, [('2007-CLtask1-target1', 'test')], None, AnnotationTransform())
        #VOCroot, [('2007-CLtask1-target2', 'test')], None, AnnotationTransform())
        #VOCroot, [('2007-CLtask1-target3', 'test')], None, AnnotationTransform())
        #VOCroot, [('2007-CLtask1-target4', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(
        #COCOroot, [('2017', 'val')], None)
            COCOroot, [('2015', 'test-dev')], None)
    else:
        print('Only VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 300
    # detector = Detect(num_classes,0,cfg)
    # detector = Detect_tf(num_classes, 0, 1000, 100, cfg, force_cpu=args.cpu)
    # detector = Detect_tf_soft(num_classes_target1, 0, cfg)
    detector = Detect_tf_soft_cls(num_classes_target1, 0, cfg)
    #detector = Detect_tf_soft_source_cls(num_classes, 0, cfg)
    save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    test_net(save_folder, net, detector, args.cuda, testset,
             BaseTransform(net.size, rgb_means, (2, 0, 1)),
             top_k, thresh=0.005)
