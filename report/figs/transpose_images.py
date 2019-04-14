import sys
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np 
import glob

print(sys.argv)

base = sys.argv[1] # path to file, ex /Users/danmacdonald/Google Drive/retina-unet-master/an_test_004/
if base[-1] == '/':
	base = base[:-1]

def convert_long_im_to_wide(img_path):
	img_ = Image.open(img_path)
	img = np.array(img_)
	im1 = img[:256,:]
	im2 = img[256:2*256,:]
	im3 = img[2*256:3*256,:]
	im4 = img[3*256:4*256,:]
	wide = np.zeros(img.T.shape)
	wide[:,:256] = im1
	wide[:,1*256:2*256] = im2
	wide[:,2*256:3*256] = im3
	wide[:,3*256:4*256] = im4
	return wide

if __name__ == '__main__':
	exp_name = base.split('retina-unet-master/')[1]
	files = glob.glob(base + '/' + exp_name+'_Original_GroundTruth_Prediction*.png')
	print('num files', len(files))
	result_dir = 'transposed'
	file_path = base+'/'+result_dir+'/'
	if not os.path.exists(file_path):
		os.makedirs(file_path)
		print('Will make dir', file_path)    
	for file in files:
		file_name = file.split(base)[1]
		print(file_name)
		wide = convert_long_im_to_wide(file)		
		img = Image.fromarray(wide)
		if img.mode != 'RGB':
			img = img.convert('RGB')
		img.save(file_path+file_name)
		# print('Saved')