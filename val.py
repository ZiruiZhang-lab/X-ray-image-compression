from model.HLFSP_Net import HLFSP
from model.ms_ssim_torch import ms_ssim
import os
import sys
import argparse
from PIL import Image
import numpy as np
from utils1 import *


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example testing script.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/0.014bpp.tar",help="Path to a checkpoint")
    parser.add_argument("--input", type=str, default="./minidataset/minidatast/test", help="Path to input_images")
    parser.add_argument("--real", action="store_true", default=False)
    args = parser.parse_args(argv)
    return args

def main(argv):
    os.makedirs("image/output_machines", exist_ok=True)
    args = parse_args(argv)
    p = 128
    path = args.input
    img_list = []
    for file in os.listdir(path):
        img_list.append(file)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    icm = HLFSP(192,  320, lamb=24000)
    icm = icm.to(device)
    icm.eval()
    Bit_rate = 0
    Psnr = 0
    sumMsssim = 0
    sumMsssimDB = 0

    dictory = {}
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        icm.load_state_dict(dictory)

    if args.real:
        print('\n::real compression::\n')
        icm.update()
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            print(img_path[-16:])
            img = Image.open(img_path).convert('L')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                out_enc = icm.compress(x_padded)
                out_dec = icm.decompress(out_enc["strings"], out_enc["shapes"])

                out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
                num_pixels = x.size(0) * x.size(2) * x.size(3)
                forward_output = icm.forward(x_padded)
                print(f'Bitrate: {(sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels):.3f}bpp')
                print(f'compute_psnr: {compute_psnr(out_dec["x_hat"], x):.3f}psnr')


                print("PSNR (forward):", compute_psnr(forward_output["x_hat"], x))


                Bit_rate += sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
                Psnr += compute_psnr(out_dec["x_hat"], x)


        print('\n---result_bpp(real)---')
        print(f'average_Bit-rate: {(Bit_rate / len(img_list)):.3f} bpp')
        print(f'average_Psnr: {(Psnr / len(img_list)):.3f} psnr')
        print('--- save image ---')
        print('compressed images are saved in "image/output_machines"\n')

    else:
        for img_name in img_list:
            img_path = os.path.join(path, img_name)
            print(img_path[-16:])
            depth_img = Image.open(img_path).convert('L')
            depth = transforms.ToTensor()(depth_img).unsqueeze(0).to(device)
            img = Image.open(img_path).convert('L')
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            x_padded, padding = pad(x, p)

            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                recon_img, mse_loss, bpp_loss, bpp_z, bpp = icm.forward(x_padded)
                print(f'Bit-rate: {bpp:.3f}bpp')

                print(f'compute_psnr: {compute_psnr(recon_img,x):.3f}psnr')

                msssim = ms_ssim(recon_img, x, data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                print(f'msssimDB: {msssimDB:.3f}msssimDB')
                print(f'msssim: {msssim:.3f}msssim')

                Bit_rate += bpp
                Psnr += compute_psnr(recon_img, x)


        print('\n---result_bpp(estimate)---')
        print(f'average_Bit-rate: {(Bit_rate / len(img_list)):.3f} bpp')
        print(f'average_Psnr: {(Psnr / len(img_list)):.3f} psnr')
        print(f'average_MsssimDB: {(sumMsssimDB / len(img_list)):.3f} MsssimDB')
        print(f'average_Msssim: {(sumMsssim / len(img_list)):.3f} Msssim')
        print('--- save image ---')
        print('compressed images are saved in "image/output_machines"\n')


if __name__ == "__main__":
    print('\n::: compress images for Machines :::\n')
    print(torch.cuda.is_available())
    main(sys.argv[1:])


