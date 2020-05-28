import torch
import utils as ut
import torchvision.transforms.functional as FT
from skimage.io import imread,imsave
from torchvision import transforms
from models import model_dict
from skimage import data,segmentation,measure,morphology,color,draw,data
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 

def apply(image_path, model_name, model_path):
    transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong() ]
                    ])

    # Load best model
    model = model_dict[model_name](n_classes=2).cuda()
    model.load_state_dict(torch.load(model_path))

    # Read Image
    image_raw = imread(image_path)
    collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
    image, _ = transformer(collection)

    batch = {"images":image[None]}

    # Make predictions
    pred_blobs = model.predict(batch, method="blobs").squeeze()
    pred_counts = int(model.predict(batch, method="counts").ravel()[0])

    # Save Output
    save_path = image_path + "_blobs_count:{}.png".format(pred_counts)

    imsave(save_path, ut.combine_image_blobs(image_raw, pred_blobs))

    #Get the centroid of output image
    center = dict()
    img = imread(save_path)
    #Save centroid
    Centorid_path = image_path + "_centroid_blobs_count:{}.png".format(pred_counts)

    #Save predicted bbox
    bbox_file = "{}_boundingbox.txt".format(image_path[:-4])
    im_name = os.path.basename(image_path)


    for i in range(pred_counts):
        props = measure.regionprops(pred_blobs)
        center[i] = props[i].centroid
        cy,cx = draw.circle(center[i][0],center[i][1],6)
        draw.set_color(img,[cy,cx],[255,0,0])
        minr,minc,maxr,maxc = props[i].bbox

        with open(bbox_file,'a+') as bf:
            name = im_name.split('_')
            y_i = int(name[1])
            tmp_x = name[2].split('.')
            x_j = int(tmp_x[0])
            bf.write(str(int(minc)+x_j)+' '+str(int(minr)+y_i)+' '+str(int(maxc)+x_j)+' '+str(int(maxr)+y_i)+'\n')


        cv2.rectangle(img,(minc,minr),(maxc,maxr),color=(255,0,0),thickness=2)
    imsave(Centorid_path,img)

    print('ok,get centroid finished')