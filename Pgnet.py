# -*- coding: utf-8 -*-
"""
@author: lsl
E-mail: cug_lsl@cug.edu.cn
"""
import sys 
import argparse
sys.path.append("/home/aistudio/code") 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import time

from Pgnet_structure import Pg_net
from Pgnet_dataset import Mydata
from loss_function import SAMLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  hyper parameters
test_batch_size = 1

parser = argparse.ArgumentParser(description='Paddle Pgnet')
# model
parser.add_argument('--model', type=str, default='Pgnet')
# dataset
parser.add_argument('--dataset', type=str, default='WV2')

# train
parser.add_argument('--in_nc', type=int, default=126, help='number of input image channels')
parser.add_argument('--endmember', type=int, default=20, help='number of endmember')
parser.add_argument('--batch_size', type=int, default=15, help='training batch size')
parser.add_argument('--num_epochs', type=int, default=500, help='number of training epochs')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--resume', type=str, default='', help='path to model checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='restart epoch number for training')
parser.add_argument('--momentum', type=float, default=0.05, help='momentum')
parser.add_argument('--step', type=int, default=100,
                    help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=100')

# test
parser.add_argument('--test', type=bool, default=False, help='test')
parser.add_argument('--load_para', type=bool, default=False, help='if load model parameters')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
opt = parser.parse_args()
print(opt)

def Pgnet(train_hs_image, train_hrpan_image, train_label,
          test_hs_image, test_hrpan_image, test_label,
          ratio=16):

    opt.in_nc = train_hs_image.shape[1]

    print(train_hs_image.shape)
    print(test_hs_image.shape)
	
    #  define data and model
    dataset0 = Mydata(train_hs_image, train_hrpan_image, train_label)
    train_loader = data.DataLoader(dataset0, num_workers=0, batch_size=opt.batch_size,
                                 shuffle=True, drop_last=True)

    dataset1 = Mydata(test_hs_image, test_hrpan_image, test_label)
    test_loader = data.DataLoader(dataset1, num_workers=0, batch_size=opt.test_batch_size,
                       shuffle=False, drop_last=False)

    model = Pg_net(band=opt.in_nc, endmember_num=opt.endmember, ratio=ratio).to(device)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    L2_loss = nn.MSELoss()
    samloss = SAMLoss()

    optimizer = optim.Adam(lr=opt.lr, params=model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.step, gamma=opt.momentum)
    for epoch in range(opt.num_epochs):
        time0 = time.time()
        loss_total = 0.0


        model.train()
        for i, (images_hs, images_pan, labels) in enumerate(train_loader):
            images_hs = images_hs.to(device, dtype=torch.float32)
            images_pan = images_pan.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            result = model(images_hs, images_pan)

            loss_l2 = L2_loss(result, labels)
            loss_sam = samloss(result, labels)

            loss = loss_l2 + 0.01*loss_sam

            loss.backward()
            optimizer.step()

            loss_total += loss.item()

        if ((epoch+1) % 10) == 0:
            print('epoch %d of %d, using time: %.2f , loss of train: %.4f' % 
                (epoch + 1, opt.num_epochs, time.time() - time0, loss_total))

        scheduler.step()
    # torch.save(model.state_dict(), 'model.pth')

    # testing model
    if opt.load_para:
        model.load_state_dict(torch.load("model.pth"))

    model.eval()
    image_all = []
    with torch.no_grad():
        for (images_hs, images_pan, _) in test_loader:
            images_hs = images_hs.to(device, dtype=torch.float32)
            images_pan = images_pan.to(device, dtype=torch.float32)

            outputs_temp = model(images_hs, images_pan)
            image_all.append(outputs_temp)
        a = torch.cat(image_all, 0)

    return a
