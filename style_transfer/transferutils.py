import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

content_patch_size, style_patch_size = 128, 128
unloader = transforms.Compose([
        transforms.ToPILImage(),  # scale imported image
        transforms.Resize((content_patch_size, content_patch_size))])  # reconvert into PIL image
def normalize(img, min_val = 0, max_val = 1):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * (max_val - min_val) + min_val

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, fig_size = 128):

    # load image
    img = Image.open(image_path)
    img = np.array(img)
    if len(img.shape)>2:
      img = img[:,:,0]
    print(img.shape)
    img = normalize(img, 0, 255)[0 : fig_size, 0 : fig_size]

    # gray to rgb for vgg

    # Expand the 1 channel grayscale to 3 channel grayscale image
    temp = np.zeros(img.shape + (3,), dtype=np.uint8)
    temp[:, :, 0] = img
    temp[:, :, 1] = img.copy()
    temp[:, :, 2] = img.copy()
    img = temp

    return Image.fromarray(img)

loader = transforms.Compose([
    transforms.Resize((128,128)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def get_result(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image[0])
    return np.asarray(image.convert('L')).astype(np.float64)

def save_result(tensor, save_path):
    np.save(save_path, get_result(tensor))