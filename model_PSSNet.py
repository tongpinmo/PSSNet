import torch.nn as nn
import torchvision
import torch
import os
from torchvision import models
from skimage import morphology as morph
from skimage import data, util
from skimage.io import imread,imsave
from skimage import data,segmentation,measure,morphology,color
import matplotlib.patches as mpatches
import numpy as np
import utils as ut
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

class BaseModel(nn.Module):
    def __init__(self, n_classes):
        super(BaseModel,self).__init__()
        self.trained_images = set()
        self.n_classes = n_classes

    @torch.no_grad()
    def predict(self, batch, method="probs"):
        if method == "counts":
            images = batch["images"].cuda()
            origin_mask = F.softmax(self(images),1)
            mask_numpy = ut.t2n(origin_mask[0])
            pred_mask = np.argmax(mask_numpy,axis=0)
            #visualize origin mask

            counts = np.zeros(self.n_classes-1)

            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue
                blobs_category = morph.label(pred_mask==category_id)
                n_blobs = (np.unique(blobs_category) != 0).sum()
                counts[category_id-1] = n_blobs
                print('counts[None]: ',counts[None])

            return counts[None]

        elif method == "blobs":

            images = batch["images"].cuda() 

            origin_mask = F.softmax(self(images),1)
            mask_numpy = ut.t2n(origin_mask[0])

            #foreground
            pred_mask = np.argmax(mask_numpy,axis=0)

            h,w = pred_mask.shape
            blobs = np.zeros((self.n_classes-1, h, w), int)


            for category_id in np.unique(pred_mask):
                if category_id == 0:
                    continue

                blobs[category_id-1] = morph.label(pred_mask==category_id)

            return blobs[None]

###----------------------------------------------------------------Unet_model-----------------------------------

def unetUp(inputs1,inputs2):
    offset_H = inputs1.size()[2] - inputs2.size()[2]
    offset_W = inputs1.size()[3] - inputs2.size()[3]
    padding = [0,offset_W,0,offset_H]
    outputs2 = F.pad(inputs2,padding)
    return torch.cat([inputs1,outputs2],1)


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)                #size is the same


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=True):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class PSSNet(BaseModel):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet50(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """
##is_deconv must be False
    def __init__(self, n_classes, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet50
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super(PSSNet,self).__init__(n_classes)
        self.num_classes = n_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet50(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512*4, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256*4 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128*4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64*4 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))


        dec5 = self.dec5(unetUp(center, conv5))
        dec4 = self.dec4(unetUp(dec5, conv4))
        dec3 = self.dec3(unetUp(dec4, conv3))
        dec2 = self.dec2(unetUp(dec3, conv2))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)


        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        x_out = nn.functional.interpolate(x_out,
                                       size=input_spatial_dim,
                                        mode="bilinear",
                                        align_corners=True)

        return x_out


class UNet16(BaseModel):
    def __init__(self, num_classes, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super(UNet16,self).__init__(num_classes)
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


