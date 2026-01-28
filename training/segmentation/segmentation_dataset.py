
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torch

class XBD_Building_Segmentation_Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        mask_path = self.data[idx][1]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        mask = tv_tensors.Mask(mask)

        img, mask = self.transform(img, mask)
        mask = (mask/255).squeeze(0).to(torch.long)

        return img, mask