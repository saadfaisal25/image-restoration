import torch
import matplotlib.pyplot as plt
import numpy as np
from model import DerainModel
from torchvision import transforms
from dataset import DerainDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # load the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = DerainDataset('data', mode='test', transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # initialize the model
    model = DerainModel()
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    criterion = torch.nn.MSELoss()

    # test the model
    model.eval()
    psnr_total = 0
    ssim_total = 0
    mse_total = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)
            loss = criterion(output, targets)
            mse_total += loss.item()
            print(f'Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

            # calculate PSNR and SSIM for each image in the batch
            output = output.cpu().numpy().transpose(0, 2, 3, 1)
            targets = targets.cpu().numpy().transpose(0, 2, 3, 1)
            psnr_batch = []
            ssim_batch = []
            for target, out in zip(targets, output):
                psnr_val = psnr(target, out, data_range=out.max() - out.min())
                ssim_val = ssim(target, out, multichannel=True, data_range=out.max() - out.min(), channel_axis=-1)
                psnr_batch.append(psnr_val)
                ssim_batch.append(ssim_val)
                psnr_total += psnr_val
                ssim_total += ssim_val

            # show the images and metrics
            inputs = inputs.cpu().numpy().transpose(0, 2, 3, 1)

            fig, axs = plt.subplots(4, 3, figsize=(12, 8))

            for j in range(4):
                axs[j, 0].imshow(inputs[j])
                axs[j, 0].set_title('Input')
                axs[j, 1].imshow(targets[j])
                axs[j, 1].set_title('Target')
                axs[j, 2].imshow(output[j])
                mse = np.mean((output[j] - targets[j]) ** 2)
                axs[j, 2].set_title(f'Output (PSNR: {psnr_batch[j]:.4f}, SSIM: {ssim_batch[j]:.4f}, MSE: {mse:.4f})')

            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.pause(0.001)
            plt.close()

    # calculate average PSNR and SSIM for all batches
    psnr_avg = psnr_total / (len(data_loader) * 4)
    ssim_avg = ssim_total / (len(data_loader) * 4)
    mse_avg = mse_total / (len(data_loader) * 4)
    print(f'Average PSNR: {psnr_avg:.4f}')
    print(f'Average SSIM: {ssim_avg:.4f}')
    print(f'Average MSE: {mse_avg:.4f}')

if __name__ == '__main__':
    test()
