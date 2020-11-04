import os
import time

import cv2
import numpy as np
from skimage import segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from time import time
import argparse

os.environ['CUDA_VISIBEL_DEVICES'] = '0'
os.chdir('/home/hchunlin/Projects/Unsupervised-Segmentation')
use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Modified Unsupervised Segmentation')
parser.add_argument('--input', metavar='I', default='image/woof.jpg', type=str, help='input image file path')
parser.add_argument('--trainEpoch', metavar='T', default=32, type=int, help='number of maximum epochs')
parser.add_argument('--minLabelNum', metavar='MIN', default=4,
                    type=int, help='if the label number small than it, break loop')
parser.add_argument('--maxLabelNum', metavar='MAX', default=64, type=int,
                    help='if the label number small than it, start to show result image.')
parser.add_argument('--mod_dim1', metavar='DIM1', default=64, type=int, help='dimention1 of MyNet')
parser.add_argument('--mod_dim2', metavar='DIM2', default=32, type=int, help='dimention2 of MyNet')
parser.add_argument('--gpuId', metavar='G', default='0', type=str, help='gpu id')
args = parser.parse_args()


# class Args(object):
#     input_image_path = args.input  # image/coral.jpg image/tiger.jpg
#     train_epoch = args.trainEpoch # 64
#     mod_dim1 = args.mod_dim1  # 64
#     mod_dim2 = args.mod_dim2  # 32
#     gpu_id = 0

#     # if the label number small than it, break loop
#     min_label_num = args.minLabelNum  # 4
#     # if the label number small than it, start to show result image.
#     max_label_num = args.maxLabelNum # 64
#     # max_label_num = 256


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


class Logger():
    def __init__(self, filepath):
        self.fileWriter = open(filepath, 'w')

    def print_log(self, *text):
        log = ''
        for t in text:
            log += str(t) + ' '
        print(log)
        self.fileWriter.write(log+'\n')

    def close_file(self):
        self.fileWriter.close()

name = os.path.split(args.input)[-1]
name = name.split('.')[0]
log_filepath = 'modify_results/report_%s.txt' % name
logger = Logger(log_filepath)

def run():
    logger.print_log("start initialization...")
    start_time0 = time()

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuId  # choose GPU:0
    image = cv2.imread(args.input)
    # image = cv2.resize(
    #     image, (int(image.shape[1]/2), int(image.shape[0]/2)), interpolation=cv2.INTER_AREA)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
    # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    start_time1 = time()
    seg_init_Time = start_time1 - start_time0
    logger.print_log('segInitTimeUsed:', seg_init_Time)

    
    '''show result of segmentation ML'''
    show = image.reshape((-1, 3)).copy()
    for inds in seg_lab:
        color_avg = np.mean(show[inds], axis=0, dtype=int)
        show[inds] = color_avg
    cv2.imwrite("felz_%s_L%02i.jpg" % (name, len(seg_lab)), show.reshape(image.shape))
    

    '''train init'''
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

    image_flatten = image.reshape((-1, 3))
    # color_avg = np.random.randint(255, size=(args.maxLabelNum, 3))
    show = image

    pytorch_init_time = time() - start_time1
    logger.print_log('PytorchInitTimeUsed:', str(pytorch_init_time))

    '''train loop'''
    start_time2 = time()
    model.train()
    for batch_idx in range(args.trainEpoch):
        '''forward'''
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        '''refine'''
        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        '''backward'''
        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        '''show image'''
        # lab_inverse: 相当于是根据拍好序的un_label对im_target进行重新编号
        # un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        un_label = np.unique(im_target)
        nLabels = len(un_label)
        if un_label.shape[0] < args.maxLabelNum and (batch_idx+1)%8==0:  # update show
            img_flatten = image_flatten.copy()
            # if len(color_avg) != un_label.shape[0]:
            #     color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]
            # for lab_id, color in enumerate(color_avg):
            #     img_flatten[lab_inverse == lab_id] = color
            # if len(color_avg) != un_label.shape[0]:
            for label in un_label:
                color_avg = np.mean(img_flatten[im_target == label], axis=0, dtype=int)
                img_flatten[im_target == label] = color_avg

            show = img_flatten.reshape(image.shape)
            cv2.imwrite("modify_results/output_%s_I%02i_L%02i.jpg" % (name, batch_idx, nLabels), show)

        # cv2.imshow("seg_pt", show)
        # cv2.waitKey(1)

        logger.print_log(batch_idx, '/', args.trainEpoch,
                         ':', nLabels, loss.item())
        if len(un_label) < args.minLabelNum:
            break

    '''save'''
    train_time = time() - start_time2
    logger.print_log('SegInit: %.2f\nPyTorchInit: %.2f\nTrainTimeUsed: %.2f' % (seg_init_Time, pytorch_init_time, train_time))
    cv2.imwrite("modify_results/final_%s_%ds.jpg" % (args.input[6:-4], train_time), show)


if __name__ == '__main__':
    run()
