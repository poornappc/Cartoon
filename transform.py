import sys
import os.path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm2d(256)
        self.norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.norm_1 = nn.BatchNorm2d(64)
        # down-convolution #
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)
        # residual blocks #
        residualBlocks = []
        for l in range(8):
            residualBlocks.append(ResidualBlock())
        self.res = nn.Sequential(*residualBlocks)
        # up-convolution #
        self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.BatchNorm2d(128)
        self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm_5 = nn.BatchNorm2d(64)
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = F.relu(self.norm_1(self.conv_1(x)))
        x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
        x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
        x = self.res(x)
        x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
        x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))
        x = self.conv_10(x)
        x = sigmoid(x)
        return x


def transform_image(input_image_path, output_image_path, model_path='cartoonGAN/generator_quantitative.pth'):
    if not (os.path.isfile(model_path)):
        raise Exception(
            'Can not find pre-trained weights file generator_quantitative.pth. Please provide within current directory.')

    checkpoint = torch.load(model_path, map_location='cpu')
    G = Generator().to('cpu')
    G.load_state_dict(checkpoint['g_state_dict'],strict=False)

    transformer = transforms.Compose([
        transforms.ToTensor()
    ])

    with Image.open(input_image_path) as img:
        pseudo_batched_img = transformer(img)
        pseudo_batched_img = pseudo_batched_img[None]
        result = G(pseudo_batched_img)
        result = transforms.ToPILImage()(result[0]).convert('RGB')
        result.save(output_image_path)


def process_frame(frame, transformer, model):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((256, 256))
    img_tensor = transformer(img)[None]
    result_tensor = model(img_tensor)
    result = transforms.ToPILImage()(result_tensor[0]).convert('RGB')
    result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    return result


def transform_video(input_video_path, output_video_path, model_path='cartoonGAN/generator_quantitative.pth'):
    if not os.path.isfile(model_path):
        print('Cannot find pre-trained weights file generator_quantitative.pth. Please provide within the current directory.')
        exit(0)

    checkpoint = torch.load(model_path, map_location='cpu')
    G = Generator().to('cpu')
    G.load_state_dict(checkpoint['g_state_dict'],strict=False)

    transformer = transforms.Compose([
        transforms.ToTensor()
    ])

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (256, 256))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_frame(frame, transformer, G)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

if __name__ == '__main__':
    input_image_path = './test.jpg'
    output_image_path = './test.jpg'
    transform_image(input_image_path, output_image_path)
