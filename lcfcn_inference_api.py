#get the centroid raw file of input_image
import torch
import utils as ut
import torchvision.transforms.functional as FT
from torchvision import transforms
from models import model_dict
from skimage import data,segmentation,measure,morphology,color,draw,io
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import os 

n_classes = 2
model_name = "ResUnet"

# Load best model
model_path = './Ablation/without_Split_mode_Global_loss/best_model_acacia_ResUnet.pth'
model = model_dict[model_name](n_classes).cuda()
model.load_state_dict(torch.load(model_path))

def lcfcn_inference(image_path):
    '''
    :param image_path:
    :return: centroid file
    '''
    transformer = ut.ComposeJoint(
                    [
                         [transforms.ToTensor(), None],
                         [transforms.Normalize(*ut.mean_std), None],
                         [None,  ut.ToLong()]
                    ])

    # Read Image
    image_raw = io.imread(image_path)
    collection = list(map(FT.to_pil_image, [image_raw, image_raw]))
    image, _ = transformer(collection)

    batch = {"images":image[None]}
    # print('batch.shape: ',image[None].shape)          #(1,3,500,500)

    # Make predictions
    pred_blobs = model.predict(batch, method="blobs").squeeze()         #(500,500)
    # print('blobs: ',pred_blobs)
    # plt.imshow(pred_blobs,cmap='gray')
    # plt.show()
    pred_counts = int(model.predict(batch, method="counts").ravel()[0])
    # print('pred_counts: ',pred_counts)

    #get the centroid of output image
    center = dict()

    #Save output,the coordinates is consistent with cutted images
    Centroid_path = image_path[:-4] + "_centroid_count:{}.png".format(pred_counts)

    #Save predicted bbox
    # bbox_file = './datasets/test_data/12months_mask2unet_bbox.txt'
    im_name = os.path.basename(image_path)


    for i in range(pred_counts):
        props = measure.regionprops(pred_blobs)            
        center[i] = props[i].centroid
        cy,cx = draw.circle(center[i][0],center[i][1],6)
        draw.set_color(image_raw,[cy,cx],[255,0,0])
        minr,minc,maxr,maxc = props[i].bbox
        
        # with open(bbox_file,'a+') as bf:
        #     name = im_name.split('_')
        #     y_i = int(name[1])
        #     tmp_x = name[2].split('.')
        #     x_j = int(tmp_x[0])
        #     bf.write(str(int(minc)+x_j)+' '+str(int(minr)+y_i)+' '+str(int(maxc)+x_j)+' '+str(int(maxr)+y_i)+'\n')

        cv2.rectangle(image_raw, (minc, minr), (maxc, maxr), color=(255, 0, 0), thickness=2)
    io.imsave(Centroid_path,image_raw)

    return center

