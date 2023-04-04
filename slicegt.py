import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import ECTransNet
from matplotlib import image as a
from PIL import Image

def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,"images", name) + ".png" for name in data]
    masks = [os.path.join(path,"masks", name) + ".png" for name in data]
    return images, masks

def load_data(path, n):
    valid_names_path = f"{path}/" + n + "_test.txt"
    valid_x, valid_y = load_names(path, valid_names_path)

    return  (valid_x, valid_y)

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    """ load_mdoel """
    image_size = 256
    size = (image_size, image_size)
    device = torch.device('cuda')
    model = ECTransNet()
    model = model.to(device)
    model.load_state_dict(torch.load('checkpoints/checkpoints.pth'))
    filename = ["clinic", "cvc300", "kvasir", "etis", "colon"]
    for n in filename:
        slicepath = "dataset/test/" + n
        savepath = "results/" + n + "/"
        """ Dataset """
        (valid_x, valid_y) = load_data(slicepath, n)

        # 获取图像名称和大小
        imagename = []
        shape = []
        for i in valid_x:
            img = Image.open(i)
            tempsize = (img.height, img.width)
            shape.append(tempsize)
            tempname = i[len(slicepath)+8:]
            imagename.append(tempname)
        data_str = f"Dataset:{n}\n Valid: {len(valid_x)}"
        print(data_str)
        valid_dataset = DATASET(valid_x, valid_y, size, transform=None)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        temp = 0
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                x = x.to(device, dtype=torch.float32)
                # y = y.to(device, dtype=torch.float32)
                y_pre = model(x)
                res = F.interpolate(y_pre, size=shape[temp], mode='bilinear', align_corners=True)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                a.imsave(savepath+imagename[temp], res, cmap='gray')
                temp = temp + 1
                jindu = round(temp/((len(valid_x)//10)*10), 4)*100
                if jindu % 10 == 0:
                    print("▓{}%".format(int(jindu)), end="")
                sys.stdout.flush()
                time.sleep(0.05)
        print("\n")
