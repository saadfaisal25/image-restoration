import os
from PIL import Image
from torch.utils.data import Dataset

class DerainDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        super(DerainDataset, self).__init__()
        self.mode = mode
        self.data_dir = root_dir
        self.transform = transform
        
        if mode == 'train':
            self.input_dir = os.path.join(root_dir, 'train', 'input')
            self.target_dir = os.path.join(root_dir, 'train', 'target')
        else:
            self.input_dir = os.path.join(root_dir, 'test', 'input')
            self.target_dir = os.path.join(root_dir, 'test', 'target')
            
        self.input_filenames = os.listdir(self.input_dir)
        self.target_filenames = os.listdir(self.target_dir)
        
    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.input_filenames[index])
        target_path = os.path.join(self.target_dir, self.target_filenames[index])
        
        input_image = Image.open(input_path)
        target_image = Image.open(target_path)
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image
    
    def __len__(self):
        return len(self.input_filenames)
