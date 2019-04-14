#==========================================================
#
#  This prepare the hdf5 datasets of the DRIVE database
#
#============================================================

import os
import h5py
import numpy as np
from PIL import Image



def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "./DATA/training/images/"
groundTruth_imgs_train = "./DATA/training/1st_manual/"
borderMasks_imgs_train = "./DATA/training/mask/"
watershed_imgs_train = "./DATA/training/watershed/"
#test
original_imgs_test = "./DATA/test/images/"
groundTruth_imgs_test = "./DATA/test/1st_manual/"
borderMasks_imgs_test = "./DATA/test/mask/"
watershed_imgs_test = "./DATA/test/watershed/"
#---------------------------------------------------------------------------------------------

Nimgs = 30
channels = 1
height = 256    # Likely smaller. check size of images
width = 256     # ditto
dataset_path = "./DATA_datasets_training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,borderMasks_dir,watersheds_dir,train_test="null"):
    imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs,height,width))
    border_masks = np.empty((Nimgs,height,width))
    watersheds = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print "original image: " +files[i]
            img = Image.open(imgs_dir+files[i]).convert('L')
            print(np.asarray(img).shape)
            imgs[i] = np.asarray(img).reshape((height,width,1))
            imgs[i] = imgs[i]/imgs[i].max()
            #corresponding ground truth
            groundTruth_name = files[i][0:4] + "_manual1.gif"
            print "ground truth name: " + groundTruth_name
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)
            #
            watershed_name = files[i][0:4] + "_watershed.gif"
            # print('watershedname',watershed_name)
            # print "watershed name: " + watershed_name
            ws_truth = Image.open(watersheds_dir + watershed_name)
            watersheds[i] = np.asarray(ws_truth)
            #corresponding border masks
            border_masks_name = ""
            if train_test=="train":
                border_masks_name = files[i][0:4] + "_mask.gif" # _training
            elif train_test=="test":
                border_masks_name = files[i][0:4] + "_mask.gif" # _test
            else:
                print "specify if train or test!!"
                exit()
            print "border masks name: " + border_masks_name
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            border_masks[i] = np.asarray(b_mask)

    print "imgs max: " +str(np.max(imgs))
    print "imgs min: " +str(np.min(imgs))
    # print('maxes',np.max(imgs),np.max(groundTruth),np.max(border_masks))
    # CHEATING
    groundTruth = 255*groundTruth/groundTruth.max()
    border_masks = 255*border_masks/border_masks.max()
    watersheds = 255*watersheds/watersheds.max()

    assert(np.max(groundTruth)==255 and np.max(border_masks)==255)
    assert(np.min(groundTruth)==0 and np.min(border_masks)==0)
    print "ground truth and border masks are correctly withih pixel value range 0-255 (black-white)"
    #reshaping for my standard tensors
    imgs = np.transpose(imgs,(0,3,1,2))
    assert(imgs.shape == (Nimgs,channels,height,width))
    groundTruth = np.reshape(groundTruth,(Nimgs,1,height,width))
    border_masks = np.reshape(border_masks,(Nimgs,1,height,width))
    watersheds = np.reshape(watersheds,(Nimgs,1,height,width))
    assert(groundTruth.shape == (Nimgs,1,height,width))
    assert(border_masks.shape == (Nimgs,1,height,width))
    return imgs, groundTruth, border_masks, watersheds

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
#getting the training datasets
imgs_train, groundTruth_train, border_masks_train, watersheds_train = get_datasets(original_imgs_train,groundTruth_imgs_train,borderMasks_imgs_train,watershed_imgs_train,"train")
print "saving train datasets"
write_hdf5(imgs_train, dataset_path + "DATA_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DATA_dataset_groundTruth_train.hdf5")
write_hdf5(border_masks_train,dataset_path + "DATA_dataset_borderMasks_train.hdf5")
write_hdf5(watersheds_train,dataset_path + "DATA_dataset_watersheds_train.hdf5")


#getting the testing datasets
imgs_test, groundTruth_test, border_masks_test, watersheds_test = get_datasets(original_imgs_test,groundTruth_imgs_test,borderMasks_imgs_test,watershed_imgs_test,"test")
print "saving test datasets"
write_hdf5(imgs_test,dataset_path + "DATA_dataset_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + "DATA_dataset_groundTruth_test.hdf5")
write_hdf5(border_masks_test,dataset_path + "DATA_dataset_borderMasks_test.hdf5")
write_hdf5(watersheds_test,dataset_path + "DATA_dataset_watersheds_test.hdf5")

