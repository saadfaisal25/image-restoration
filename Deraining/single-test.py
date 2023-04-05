import argparse
import cv2
import numpy as np
import torch
from model import DerainModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input image')
# parser.add_argument('--weights', type=str, required=True, help='Path to the saved model weights')

args = parser.parse_args()

# load input image
input = cv2.imread(args.input)
input = input.astype(np.float32) / 255.0
input = cv2.resize(input, (256, 256))
input = torch.from_numpy(input).permute(2,0,1).unsqueeze(0)

model = DerainModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
with torch.no_grad():
    output = model(input)

# Calculate PSNR, SSIM, and MSE
# psnr_val = psnr(input, output, data_range=output.max() - output.min())
# ssim_val = ssim(input, output, multichannel=True, data_range=output.max() - output.min(), channel_axis=-1)
# mse_val = ((input - output) ** 2).mean()

# transpose and concatenate images
input = np.transpose(input.squeeze(0).cpu().numpy(), (1,2,0))
output = np.transpose(output.squeeze(0).cpu().numpy(), (1,2,0))
output = np.concatenate((input, output), axis=1)
cv2.imshow('Output', output)
cv2.waitKey(0)


