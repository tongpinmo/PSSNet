#coding:utf-8
import torch, os
import torch.utils.data as data

import numpy as np
import torchvision.transforms.functional as FT
from PIL import Image
import utils as ut

#----------------------------------------------first step:get pointDict---------------------
def get_pointDict(path, imgNames):
    pointDict = {}
    for i, img_name in enumerate(imgNames):
        print(i, img_name)
        pointDict[img_name] = []
        point_path = "%s/IMG_point/" % path
        with open(point_path + img_name + '.txt','r') as tf:
            lines = tf.readlines()
            for i in lines:
            # Get the center points
                i = i.strip().split(' ')
                x = i[0]
                y = i[1]
                class_name = 'acacia'
                pointDict[img_name] += [{"x": x, "y": y, "name": class_name}]
                print('pointDict:%s: '% (img_name), pointDict[img_name])

    return pointDict



class Acacia(data.Dataset):
    name2class = {
                   "acacia":0
                }

    def __init__(self,
                 split=None,
                 transform_function=None):

        self.path  = "/mnt/a409/users/tongpinmo/projects/LCFCN/datasets/acacia-train"
        self.transform_function = transform_function

        fname_path = self.path
        path_pointJSON = "%s/pointDict.json" % self.path

        if split == "train":
            self.imgNames = [t.replace(".jpg\n", "")
                             for t in
                             ut.read_text(fname_path + "/train.txt")]


        # elif split == "val":
        #     self.imgNames = [t.replace(".jpg \n", "")
        #                      for t in
        #                      ut.read_text(fname_path + "/train_val.txt")]
        # elif split == "test":
        #     self.imgNames = [t.replace("\n","")
        #                         for t in
        #                         ut.read_text(fname_path + "/test.txt")]

        if os.path.exists(path_pointJSON):
            self.pointsJSON = ut.load_json(path_pointJSON)

        else:
            pointDict = get_pointDict(self.path, self.imgNames)
            ut.save_json(path_pointJSON, pointDict)

        self.split = split
        self.n_classes = 2

    def __len__(self):
        return len(self.imgNames)

    def __getitem__(self, index):
        img_name = self.imgNames[index]

        path_Acacia = self.path
        img_path = path_Acacia + "/IMG_Palm/%s.jpg" % img_name
        img = Image.open(img_path).convert('RGB')

        # GET POINTS
        w, h = img.size
        points = np.zeros((h, w, 1))
        counts = np.zeros(1)
        counts_difficult = np.zeros(1)

        if self.split == "train":
            pointLocs = self.pointsJSON[img_name]

            for p in pointLocs:
                if int(p["x"]) > w or int(p["y"]) > h:
                    continue
                else:
                    points[int(p["y"]), int(p["x"])] = self.name2class[p["name"]] + 1
                    counts[self.name2class[p["name"]]] += 1
                    counts_difficult[self.name2class[p["name"]]] += 1
        #where there is pixels,there is points
        points = FT.to_pil_image(points.astype("uint8"))

        if self.transform_function is not None:
            img, points = self.transform_function([img, points])

        return {"counts": torch.LongTensor(counts),
                "images": img, "points": points,
                "image_path": img_path,
                "index": index, "name": img_name}
