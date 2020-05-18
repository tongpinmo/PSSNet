import torch
import torch.nn.functional as F
import numpy as np 
from skimage.morphology import watershed
from skimage.segmentation import find_boundaries
from scipy import ndimage
import utils as ut
import cv2
import os
import matplotlib.pyplot as plt

def lc_loss(model, batch, epoch):
    
    model.train()
    N = batch["images"].size(0)                             #size:(1,3,500,500)
    assert N == 1

    blob_dict = get_blob_dict(model, batch)
     
    # print('blob_dict: ',blob_dict)

    # put variables in cuda
    images = batch["images"].cuda()
    points = batch["points"].cuda()                         #torch.cuda().LongTensor()
    # print('points.shape: ',points.shape)
    counts = batch["counts"].cuda()
    #print(images.shape)
    name = batch["name"][0]
    # print('name: ', name)
    epoch = str(epoch)

    # save the predicted blobs
    blobs = blob_dict["blobs"]
    # plt.imsave(os.path.join('figures/connect_1_blobs/' + name + '_' + epoch + '_epoch_blobs' + '.jpg'), blobs.squeeze(),cmap='gray')
    # print('get predicted blobs')


    O = model(images)
    # print('O:',O)
    # print('O.shape: ',O.shape)               qunalile   #torch.Size([1, 2, 500, 500])
    S = F.softmax(O, 1)                           #softmax in class channel
    # print('S:',S)
    # print('S.shape: ',S.shape)                  #S.shape:  torch.Size([1, 2, 500, 500])

    #verify the channel of class
    S_numpy = ut.t2n(S[0])                        #(2,500,500)
    # probs = S_numpy[1]
    # print('probs: ',probs)

    S_log = F.log_softmax(O, 1)
    # print('S_log:', S_log)
    # print('S_log.shape:',S_log.shape)

    # POINT LOSS
    loss = F.nll_loss(S_log, points,
                       ignore_index=0,
                       reduction='sum')
    # FP loss
    if blob_dict["n_fp"] > 0:
        loss += compute_fp_loss(S_log, blob_dict)

    #SPLIT LOSS

    # Split_mode loss
    if blob_dict["n_multi"] > 0:
       loss += compute_split_loss(S_log, S, points, blob_dict)

    # Global loss
    S_npy = ut.t2n(S.squeeze())
    points_npy = ut.t2n(points).squeeze()
    for l in range(1, S.shape[1]):
        points_class = (points_npy==l).astype(int)

        if points_class.sum() == 0:
            continue

        T = watersplit(S_npy[l], points_class)
        #Fixme:why 1-T?
        T = 1 - T
        scale = float(counts.sum())
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')

    # Add to trained images
    model.trained_images.add(batch["image_path"][0])

    return loss / N


# Loss Utils
def compute_image_loss(S, Counts):
    n,k,h,w = S.size()                                      #torch.Size[1,2,500,500]

    # GET TARGET
    ones = torch.ones(Counts.size(0), 1).long().cuda()      #[[1]]
    BgFgCounts = torch.cat([ones, Counts], 1)               #torch.Size([1, 2])
    Target = (BgFgCounts.view(n*k).view(-1) > 0).view(-1).float()          #tensor([1., 1.], device='cuda:0')todo? view(-1)>0

    # print('ones:',ones)
    # print('ones.shape:',ones.shape)
    # print('BgFgCounts: ',BgFgCounts)
    # print('BgFgCounts.shape: ',BgFgCounts.shape)
    # print('Target:',Target)
    # print('Target.shape: ',Target.shape)

    # GET INPUT
    Smax = torch.max(S.view(n,k,h*w),dim=2)[0].view(-1)                            #tensor([1,1])
    # print('torch.max(2):',torch.max(S.view(n,k,h*w),dim=2))
    b = torch.max(S.view(n,k,h*w),dim=2)[0]
    # print('b:', b)

    loss = F.binary_cross_entropy_with_logits(Smax, Target, reduction='sum')

    return loss

def compute_fp_loss(S_log, blob_dict):
    blobs = blob_dict["blobs"]             #predicted blobs (1,500,500)

    scale = 1.
    loss = 0.

    for b in blob_dict["blobList"]:
        if b["n_points"] != 0:              #attention to background,foreground is ignored
            continue
        #when b["n_points"]==0 do
        T = np.ones(blobs.shape[-2:])
        # print('T.shape: ',T.shape)
        T[blobs[b["class"]] == b["label"]] = 0  #find the fp class,and let it to be 0
        # print('T: ',T.shape) 
        loss += scale * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')
    return loss

#todo delete epoch
def compute_split_loss(S_log, S, points, blob_dict):
    blobs = blob_dict["blobs"]
    #save big blobs
    # plt.imsave(os.path.join('figures/multi_blobs/'+ name +'_'+epoch+'_epoch_blobs'+'.jpg'),blobs.squeeze())

    S_numpy = ut.t2n(S[0])
    points_numpy = ut.t2n(points).squeeze()

    loss = 0.

    for b in blob_dict["blobList"]:
        # print('b: ',b)
        if b["n_points"] < 2:
            continue
        #only when the multi blobs do
        l = b["class"] + 1                              #l=1
        probs = S_numpy[b["class"] + 1]                 #the probability belongs to tree

        points_class = (points_numpy==l).astype("int")
        blob_ind = blobs[b["class"]] == b["label"]          #class=0

        T = watersplit(probs, points_class*blob_ind)*blob_ind
        T = 1 - T
        # print('T: ',T)
        # print('T.shape: ', T.shape)
        # plt.imsave(os.path.join('figures/boundaries/' + name + '_' + epoch + '_epoch_T' + '.jpg'), T.squeeze(),cmap='gray')

        scale = b["n_points"] + 1       #fixme:why +1
        loss += float(scale) * F.nll_loss(S_log, torch.LongTensor(T).cuda()[None],
                        ignore_index=1, reduction='elementwise_mean')

    return loss


def watersplit(_probs, _points):
    points = _points.copy()

    points[points!=0] = np.arange(1, points.sum()+1)
    points = points.astype(float)

    probs = ndimage.black_tophat(_probs.copy(), 7)
    # plt.imsave(os.path.join('figures/probs/'+name+'_'+epoch+'_epoch_probs.jpg'),probs,cmap='gray')
    seg = watershed(probs, points)

    return find_boundaries(seg)


@torch.no_grad()
def get_blob_dict(model, batch, training=False): 
    blobs = model.predict(batch, method="blobs").squeeze()              #morph.label
    # print('blobs:',blobs)
    # print('blobs.shape: ',blobs.shape)                          # (500, 500)

    points = ut.t2n(batch["points"]).squeeze()                    #1 with annotations,0 is background

    if blobs.ndim == 2:
        blobs = blobs[None]

    blobList = []

    n_multi = 0
    n_single = 0
    n_fp = 0
    total_size = 0

    #print('blobs.shape: ',blobs.shape)                  #blobs.shape:  (1, 500, 500)
    
    for l in range(blobs.shape[0]):                     #only two kinds so :l=0
        class_blobs = blobs[l]                          #blob of every kind
        points_mask = points == (l+1)                   #select the mask according to the class,select the tree
        # print('points == (l+1): ',points == (l+1))

        # print('class_blobs: ',class_blobs)
        # print('points_mask: ',points_mask)

        # Intersecting:select the regions with point annotations
        blob_uniques, blob_counts = np.unique(class_blobs * (points_mask), return_counts=True)
        # print('np.unique(class_blobs * (points_mask), return_counts=True): ',np.unique(class_blobs * (points_mask), return_counts=True))
        # print('blob_uniques: ',blob_uniques)
        # print('blob_counts: ',blob_counts)              #the first number which is so large because of belonging to background?

        #FP class
        uniques = np.delete(np.unique(class_blobs), blob_uniques)
        # print('np.unique(class_blobs): ',np.unique(class_blobs))
        # print('uniques: ',uniques)

        for u in uniques:
            blobList += [{"class":l, "label":u, "n_points":0, "size":0,
                         "pointsList":[]}]
            n_fp += 1

        #with points annotations
        #solve multi blobs situation
        for i, u in enumerate(blob_uniques):
            if u == 0:          #stands for background
                continue

            pointsList = []
            blob_ind = class_blobs==u				#select the tree class with point-annotations
            # print('blob_ind: ',blob_ind)

            locs = np.where(blob_ind * (points_mask))
            # print('locs: ',locs)
            # print('locs[0]:',locs[0])

            for j in range(locs[0].shape[0]):
                pointsList += [{"y":locs[0][j], "x":locs[1][j]}]    
            # print('pointsList: ',pointsList)
            
            assert len(pointsList) == blob_counts[i]

            #single blob
            if blob_counts[i] == 1:
                n_single += 1
            else:
                n_multi += 1
            size = blob_ind.sum()
            total_size += size
            #save the multi blobs
            blobList += [{"class":l, "size":size, 
                          "label":u, "n_points":blob_counts[i],
                          "pointsList":pointsList}]
            

    blob_dict = {"blobs":blobs, "blobList":blobList, 
                 "n_fp":n_fp, 
                 "n_single":n_single,
                 "n_multi":n_multi,
                 "total_size":total_size}

    return blob_dict
