import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HideImage(Dataset):
    def __init__(self, root_path, im_size, transform=None):
        self.cover = None
        self.im_size = im_size
        self.root_path = root_path
        self.transform = transform if transform else transforms.Compose([
            transforms.CenterCrop(im_size),
            transforms.ToTensor()
        ])
        self.load_images()

    def load_images(self):
        img_list = [os.path.join(self.root_path, img_name) for img_name in os.listdir(self.root_path)]
        self.cover = img_list

    def __len__(self):
        return len(self.cover)

    def __getitem__(self, idx):
        cover_path = self.cover[idx]
        cover = Image.open(cover_path).convert('RGB')
        cover = torch.round(self.transform(cover) * 255.)
        secret = torch.randint(0, 2, size=cover.shape) / 1.
        return cover, secret
