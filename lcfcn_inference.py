import argparse
import sys,cv2
import ctypes
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import torch
import argparse
import experiments, train, test, summary
from timer import Timer
from models import model_dict

import lcfcn_inference_api

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

lib_input_op = ctypes.cdll.LoadLibrary
lib_so = lib_input_op("./prediction_centroid/prediction_centroid.so")


#-------------------------------- First Step:cut image---------------------------
img_size_h = 500
img_size_w = 500
offset = 200
ImgPath_input = "./datasets/test_data/6Months_Crop.jpg"
ResName_input = "./figures/Acacia_infer_cut"


if os.path.exists(ResName_input):
    shutil.rmtree(ResName_input)
os.makedirs(ResName_input)


ResName_input_ = os.path.join(ResName_input, "img_")
ImgPath = bytes(ImgPath_input.encode('utf-8'))
ResName = bytes(ResName_input_.encode('utf-8'))

lib_so.cut_img(ImgPath, ResName, img_size_h, img_size_w, offset)
print("Ok,CUT IMAGE DONE...")
#
#------------------------------Second Step:Prediction with NET----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',dest='exp_name', default="trancos")
    parser.add_argument('-image_path',dest='image_path', default=None)
    parser.add_argument('-model_name',dest='model_name', default=None)
    parser.add_argument('-r', '--reset', action="store_const", const=True, default=False, help="If set, a new model will be created, overwriting any previous version.")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    centroid_file = "{}_prediction_centroid_raw.txt".format(ImgPath_input[:-4])

    #predict
    test_img_dir = ResName_input
    im_names = []
    for per_file in os.listdir(test_img_dir):
        im_names.append(per_file)

    timer = Timer()
    timer.tic()

    if os.path.exists(centroid_file):
        os.remove(centroid_file)

    for im_name in im_names:
        args.image_path = os.path.join(test_img_dir,im_name)

        center = lcfcn_inference_api.lcfcn_inference(args.image_path)

        # Save prediction_centroid_raw_file
        # the coordinates must be consistent with origin IMAGE

        with open(centroid_file, 'a+') as tf:
            for key in center.keys():
                value = center[key]
                #(x+j,y+i)
                name = im_name.split('_')
                y_i = int(name[1])
                tmp_x = name[2].split('.')
                x_j = int(tmp_x[0])

                tf.writelines([str(int(value[1]+x_j)), ' ', str(int(value[0]+y_i))])
                tf.write('\n')

    print('ok,get centroid raw files')


# ----------------------------Third step:Comb-Fusion----------------------------------
#   raw_prediction_centroid file 2 fused_index file

    test_file = centroid_file
    raw_index = bytes(test_file.encode('utf-8'))
    img_input = ImgPath

    dis_comb = 40
    fused_index_ = ImgPath_input[:-4] + "_prediction_index_fused_dis_comb_{}.txt".format(dis_comb)
    fused_index = bytes(fused_index_.encode('utf-8'))
    img_output_ = ImgPath_input[:-4] + "_prediction_dis_comb_{}.jpg".format(dis_comb)
    img_output = bytes(img_output_.encode('utf-8'))


    lib_so.comb_xy(raw_index, img_input, fused_index, img_output, dis_comb)
    print("FUSED INDEX DONE...")


    timer.toc()
    print(('Detection took {:.3f}s').format(timer.total_time))

#----------------------------Forth step:Compare-label-&-predicted-----------------------------------------------
    #
    dis_compare = 40
    label_file = './datasets/test_data/06Months.txt'
    label_file = bytes(label_file.encode('utf-8'))
    index_fused_file = fused_index
    lib_so.compare_center(label_file,index_fused_file,dis_compare)

    print('ok ,compare finished')



















