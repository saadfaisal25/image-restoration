import torch
from model import DerainModel
from torchvision import transforms
from dataset import DerainDataset
from torch.utils.data import DataLoader

def train():
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
    dataset = DerainDataset('data', mode='train', transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # initialize the model
    model = DerainModel()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # train the model
    for epoch in range(10):
        for i, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{10}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    # save the model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    train()
