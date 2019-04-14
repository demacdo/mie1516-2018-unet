from PIL import Image
import os

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
	base = '/Users/danmacdonald/Google Drive/retina-unet-master/'
	exp_name = 'an_test_004/'
	file_name = 'an_test_004_Original_GroundTruth_Prediction7.png'
	img_path = base + exp_name + file_name
	wide = convert_long_im_to_wide(img_path)
	plt.imshow(wide)
	result_dir = 'transposed/'
	if not os.path.exists(base+exp_name+result_dir):
    	# os.makedirs(base+exp_name+result_dir)
    	print('Will make dir and save file:', base+name_experiment+result_dir+file_name)    
  		# wide.save(file + ".thumbnail", "JPEG")