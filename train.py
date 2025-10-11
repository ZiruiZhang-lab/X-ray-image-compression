# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from Meter import AverageMeter
import os
import torchvision
from MSSSIM import MSSSIM

from utils.torch_utils import intersect_dicts
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler

from model.HLFSP_Net import HLFSP

from utils.image import ImageFolder
from torchvision import transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warmup_step = 0
cur_lr = 0.0001
print_freq = 50
global_step = 0
decay_interval = 1000
lr_decay = 0.1

class Trainer:
    def __init__(self, trainloader, testloader, model, args):
        self.model = model.cuda()
        self.loss_func = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.trainloader = trainloader
        self.testloader = testloader
        self.optim = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.psnr, self.mse, self.msssim, self.bpp = [AverageMeter(print_freq) for _ in range(4)]
        self.psnr_val, self.mse_val, self.msssim_val, self.bpp_val = [AverageMeter(print_freq) for _ in range(4)]
        self.best_psnr = 0.
        self.max_bpp = 100
        self.args = args
        self.MSSSIM = MSSSIM()
        self.global_step = global_step
        self.scheduler = LR_Scheduler('step', num_epochs=args.epoch, base_lr=args.lr, lr_step=args.step)

    def adjust_learning_rate(self, optimizer):
        global cur_lr
        global warmup_step
        if self.global_step < warmup_step:
            lr = self.args.lr * global_step / warmup_step
        elif self.global_step < decay_interval:
            lr = self.args.lr
        else:
            lr = self.args.lr * (lr_decay ** (global_step // decay_interval))
        cur_lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def Fourier_frequency_loss(img, recon_img, criterion):

        label_fft3 = torch.fft.fft2(img, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

        pred_fft3 = torch.fft.fft2(recon_img, dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

        f3 = criterion(pred_fft3, label_fft3)
        loss_fft = f3
        return loss_fft


    def train(self, epoch, total_epoch):
        self.model.train()
        with tqdm(total=len(self.trainloader)) as tbar:
            for i, data in enumerate(self.trainloader):

                self.global_step += 1

                img = data.cuda()
                recon_img, mse_loss, bpp_loss, bpp_z, bpp = self.model(img)
                self.scheduler(self.optim, i, epoch, self.best_psnr)

                psnr = -10 * torch.log10(self.loss_func(img, recon_img))

                sim = self.MSSSIM(img, recon_img)


                loss = mse_loss + bpp_loss

                self.optim.zero_grad()
                loss.backward()

                def clip_gradient(optimizer, grad_clip):
                    for group in optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad.data.clamp_(-grad_clip, grad_clip)

                clip_gradient(self.optim, 5)

                self.optim.step()
                self.mse.update(loss.item())
                self.psnr.update(psnr.item())
                self.msssim.update(sim.item())
                self.bpp.update(bpp.item())


                tbar.set_description(
                    'Train loss: %.3f  PSNR: %.3f MSSSIM: %.3f bpp: %.3f' % (self.mse.avg, self.psnr.avg,
                                                                             self.msssim.avg, self.bpp.avg,))
                tbar.update()

    def test(self, epoch):
        self.model.eval()

        with tqdm(total=len(self.testloader)) as tbar:
            for i, data in enumerate(self.testloader):
                img = data.cuda()
                with torch.no_grad():
                    recon_img, mse_loss, bpp_loss, bpp_z, bpp = self.model(img)

                loss = mse_loss + bpp_loss

                psnr = -10 * torch.log10(self.loss_func(img, recon_img))

                sim = self.MSSSIM(img, recon_img)
                self.mse_val.update(loss.item())
                self.psnr_val.update(psnr.item())
                self.msssim_val.update(sim.item())
                self.bpp_val.update(bpp.item())

                tbar.set_description('Test loss: %.3f  PSNR: %.3f MSSSIM: %.3f bpp: %.3f' % (
                    self.mse_val.avg, self.psnr_val.avg, self.msssim_val.avg, self.bpp_val.avg))
                tbar.update()

        self.save_model(self.args.save_images, epoch)


        if self.psnr_val.avg > self.best_psnr:
            self.best_psnr = self.psnr_val.avg

            if self.args.vis:
                for b in range(recon_img.shape[0]):
                    reimg = recon_img[b][:3, :, :].cpu()
                    reimg = torchvision.transforms.functional.to_pil_image(reimg)
                    origin = img[b][:3, :, :].cpu()
                    origin = torchvision.transforms.functional.to_pil_image(origin)

                    reimg.save('{}/{}_{}_recon.png'.format(self.args.save_images, str(b), str(b)))
                    origin.save('{}/{}_{}_ori.png'.format(self.args.save_images, str(b), str(b)))

            self.save_model(self.args.save_images,epoch, is_best=True)
            print('-** weight saved in ', self.args.save_images)


    def save_model(self, name, epoch=None, is_best=False):
        if not os.path.exists(name):
            os.mkdir(name)

        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optim.state_dict()
        }, os.path.join(name, "latest.pth.tar"))

        if epoch % 200 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optim.state_dict()
            }, os.path.join(name, f"{epoch}.pth.tar"))

        if is_best:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optim.state_dict()
            }, os.path.join(name, "best.pth.tar"))




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--pretrained", default="")
    parser.add_argument("--save_images", default="./checkpoint/26")

    parser.add_argument('--lr', default=0.0001, type=int, help='initial learning rate for model training')
    parser.add_argument('--epoch', default=600, type=int, help='model training epoch')
    parser.add_argument('--startepoch', default=0, type=int, help='model training epoch')
    parser.add_argument('--step', default=200, type=int, help='model training step')
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
    parser.add_argument('--out_channel_N', default=192, type=int, help='out_channel_N')
    parser.add_argument('--out_channel_M', default=320, type=int, help='out_channel_M')
    parser.add_argument("--dataset", default="./dataset",  help='training dataset path')
    parser.add_argument("--testpath", default="./dataset",  help='validation dataset path')
    parser.add_argument('--vis', default=True, type=bool, help='print image')
    parser.add_argument('--test', default=False, type=bool, help='model test')
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    args = parser.parse_args()

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_set = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_set = ImageFolder(args.dataset, split="test", transform=test_transforms)

    trainloader = DataLoader(train_set, batch_size=args.batchsize, num_workers=2, shuffle=True)

    testloader = DataLoader(test_set, batch_size=args.batchsize, num_workers=2, shuffle=False)
    torch.backends.cudnn.enabled = True

    if args.save_images != '':
        os.makedirs(args.save_images, exist_ok=True)

    out_channel_N, out_channel_M = args.out_channel_N, args.out_channel_M
    model = HLFSP(out_channel_N, out_channel_M, lamb=24000)


    train_val = Trainer(trainloader, testloader, model, args)

    start_epoch = args.startepoch
    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        state_dict = intersect_dicts(checkpoint['state_dict'], model.state_dict())
        model.load_state_dict(state_dict, strict=args.test)

        start_epoch = args.startepoch

    else:
        start_epoch = args.startepoch

    print('Start ...')

    if args.test:
        train_val.test(1)
    else:
        for epoch in range(start_epoch, args.epoch):
            train_val.train(epoch, args.epoch)
            train_val.test(epoch)
