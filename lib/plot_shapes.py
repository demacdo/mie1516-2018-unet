import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)

from scipy import ndimage as ndi 
from skimage import transform as tf
from skimage.morphology import watershed, disk, square, erosion
from skimage.feature import peak_local_max
from skimage.filters import rank, sobel
from skimage.util import img_as_ubyte
from skimage.transform import rescale

from skimage import io 
import glob
import os
plt.ion()
from matplotlib.widgets import Slider, Button, RadioButtons
plt.rcParams['image.cmap'] = 'gray'

io.use_plugin('pil')

xx = 256
yy = 256

def create_blank(x_len,y_len):
    img = np.zeros((x_len, y_len), dtype=np.double)
    return img

# def resize(image):
#   image_rescaled = rescale(image, 1.0 / 8.0, anti_aliasing=True)
#   return image_rescaled

def generate_truth(blank,params):
    sac = create_sac(blank,params)
    parent = create_parent(blank,params)
    complexity = create_complexity(blank,params)
    sac = transform_sac(sac,params)
    parent = transform_parent(parent)
    complexity = transform_complexity(complexity,params)
    check = np.logical_and(sac, parent).astype(float)
    # print((check==True).any())
    if (check==True).any() != True:
        print('rollin')
        sac = np.roll(sac, 10, axis=0) # down

    check2 = np.logical_and(complexity,parent).astype(float)
    if (check==True).any() != True:
        print('rollin')
        complexity = np.roll(complexity, -10, axis=0) # down
    truth = np.logical_or(sac, parent).astype(float) # (sac + parent)/2.0 # 
    truth = np.logical_or(truth,complexity).astype(float)
    # smooth = ndi.gaussian_filter(truth, sigma=1.0)
    # truth = smooth
    truth = rank.median(truth, disk(3))
    truth = truth/truth.max()
    # print('truth',truth[0])
    # print('TRUTH:',truth.max(),truth.min())
    truth_full = truth.copy()
    truth_small = rescale(truth, 1.0 / (xx/32.0), anti_aliasing=True)

    return truth_small, truth_full

def create_sac(img,params):
    ellipse_ratio = params['sac_ellipse']
    vert = params['sac_vertical']
    size = params['sac_size']

    img = img.copy()
    x = img.shape[0]
    y = img.shape[1]
    a = x*size           # Reference size - 0.1 of img shape
    b = ellipse_ratio*a # width
    x_origin = x*0.5 # Position at centre
    y_origin = y*0.5 - (a*vert) # Position just above centre prop to radius 
    rr, cc = ellipse(y_origin,x_origin,a,b,img.shape)
    img[rr, cc] = 1
    return img

def create_complexity(img,params):
    ellipse_ratio = params['complex_ellipse']
    vert = params['complex_vertical']
    size = params['complex_size']

    img = img.copy()
    x = img.shape[0]
    y = img.shape[1]
    b = x*size           # Reference size - 0.1 of img shape
    a = ellipse_ratio*b # width

    x_origin = x*0.5 # Position at centre
    y_origin = y*0.5 + (a*vert) # Position just above centre prop to radius 
    rr, cc = ellipse(y_origin,x_origin,a,b,img.shape)
    img[rr, cc] = 1
    return img

def create_parent(img,params):
    ellipse_ratio = params['par_ellipse']
    vert = params['sac_vertical']
    size = params['sac_size']

    img = img.copy()
    x = img.shape[0]
    y = img.shape[1]
    a = y*0.04
    b = a*ellipse_ratio

    x_origin = x*0.5
    y_origin = y*0.5 #- (x*size*vert) 
    rr, cc = ellipse(y_origin,x_origin,a,b,img.shape) #, orientation=math.pi / 4.)
    img[rr, cc] = 1
    return img


def transform_sac(image,params):
    # Undulation index - sine transform in x and y position
    # Aspect ratio - varying a and b of ellipse
    # Ellipticity index - ?
    # Conicity parameter - how conelike
    # Bottleneck factor - how pinched the neck is
    freq = params['und_freq_sac']
    amp = params['und_amp_sac']
    vert = params['sac_vertical']

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] + np.sin(np.linspace(0, freq[0] * np.pi, src.shape[0])) * amp[0]
    dst_cols = src[:, 0] + np.sin(np.linspace(0, freq[1] * np.pi, src.shape[0])) * amp[1]
    # dst_rows *= 1.5
    # dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    # Transform
    tform = tf.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = tf.warp(image, tform, output_shape=(rows, cols))
    return out

def transform_complexity(image,params):
    # Undulation index - sine transform in x and y position
    # Aspect ratio - varying a and b of ellipse
    # Ellipticity index - ?
    # Conicity parameter - how conelike
    # Bottleneck factor - how pinched the neck is
    freq = params['und_freq_complex']
    amp = params['und_amp_copmlex']

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] + np.sin(np.linspace(0, freq[0] * np.pi, src.shape[0])) * amp[0]
    dst_cols = src[:, 0] + np.sin(np.linspace(0, freq[1] * np.pi, src.shape[0])) * amp[1]

    dst = np.vstack([dst_cols, dst_rows]).T

    # Transform
    tform = tf.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = tf.warp(image, tform, output_shape=(rows, cols))
    img = out
    # New
    rot = None #np.random.uniform(0,0.25*np.pi) # Random angle between 0 and 1 rad
    shear = np.random.uniform(-0.2,0.2) # Random shear between 0 and 0.5 rad
    # Shifting to zero, rotating, and shifting back
    shift_y, shift_x = (np.array(img.shape)-1) / 2.
    tf_rotate = tf.SimilarityTransform(rotation=rot)
    tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
    tform = tf.AffineTransform(scale=None, rotation=None, shear=shear, translation=None) 
    #https://stackoverflow.com/questions/25895587/python-skimage-transform-affinetransform-rotation-center
    # img = tf.warp(img, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)
    # mask = transform.warp(mask, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)
   # shear in rad
    out = tf.warp(img, tform.inverse, output_shape=img.shape)
    # mask = warp(mask, tform.inverse, output_shape=mask.shape)

    return out

def pad(num):
    if num < 10: 
        out = '000'+ str(num)
    elif num <100:
        out = '00' + str(num)
    elif num < 1000:
        out = '0' + str(num)
    return out

def transform_parent(image):
    freq = params['und_freq_par']
    amp = params['und_amp_par']

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, freq[0] * np.pi, src.shape[0])) * amp[0]
    dst_cols = src[:, 0] - np.sin(np.linspace(0, freq[1] * np.pi, src.shape[0])) * amp[1]
    # dst_rows *= 1.5
    # dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T

    # Transform
    tform = tf.PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out = tf.warp(image, tform, output_shape=(rows, cols))
    return out

def blur_image(img):
    rand = np.random.normal(1.0,0.05)
    sigma = rand*round(img.shape[0]/50.0,0)
    print(sigma)
    blurred = ndi.gaussian_filter(img, sigma=sigma)
    blurred = np.clip(blurred,0.0,1.0)
    blurred = blurred/blurred.max()
    return blurred

def show_images(truth,blurred,distance,watershed_seg):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6,6),num=1)#, sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(truth)
    ax[1].imshow(blurred)
    ax[2].imshow(distance)
    ax[3].imshow(watershed_seg)
    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()

def test_image():
    x, y = np.indices((80, 80)) 
    x1, y1, x2, y2 = 28, 28, 44, 52 
    r1, r2 = 16, 20 
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2 
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2 
    image = np.logical_or(mask_circle1, mask_circle2) 
    return image

def watershed_segmentation(image):
    # distance = ndi.distance_transform_edt(image)
    # local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3))) #labels=image)
    # markers = ndi.label(local_maxi)[0]
    # labels = watershed(-distance, markers, mask=image)
    # labels = watershed(image,1)

    image = img_as_ubyte(image)
    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]
    # markers = markers/markers.max()
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))
    # gradient = gradient/gradient.max()
    # gradient = sobel(image)
    erode = generate_mask(image,gradient)
    # process the watershed
    labels = watershed(gradient, markers)
    labels -=1
    labels = labels.astype(bool)
    print('LABELS',labels.max(),labels.min())
    # labels[labels!=0] = 1

    return gradient, labels, erode

def generate_mask(image,gradient):
    erode = erosion(gradient, square(5))
    mask = np.logical_or(image, gradient)
    return mask

def plot_overlay(truth,blur,segmented):
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
    # ax = axes.ravel()
    img = np.zeros((truth.shape[0], truth.shape[1], 3), dtype=np.double)
    img[:,:,0] = segmented
    # img[:,:,2] = blur
    img[:,:,1] = truth
    axes.imshow(img)
    return img
 
def remove_old_figures():
    base = '/Users/danmacdonald/Google Drive/retina-unet-master/DATA/'
    dirs = ['training/1st_manual/','training/images/','training/mask/','test/1st_manual/','test/images/','test/mask/','training/watershed/','test/watershed/', 'training/downsampled/','test/downsampled/']
    for d in dirs:
        files = glob.glob(base + d + '*')
        for f in files:
            os.remove(f)
    for d in dirs:
        files = glob.glob(base + d + '.*')
        for f in files:
            os.remove(f)

def save_figures(truth,blurred,mask,watershed,downsampled,idx=0,split=0.5):
    
    # endings: DATA, DATA_small_input, DATA_small2large_input, DATA_complex
    base = '/Users/danmacdonald/Google Drive/retina-unet-master/DATA/'
    idy = pad(idx)
    truth = 255*truth/truth.max()
    blurred = 255*blurred/blurred.max()
    mask = 255*mask/mask.max()
    watershed = 255*watershed/watershed.max()

    truth = truth.astype(int)
    blurred = blurred.astype(int)
    mask = mask.astype(int)
    watershed = watershed.astype(int)

    print('WSMM', watershed.max(), watershed.min())

    c1 = truth.max() == 255 and truth.min() == 0
    c2 = blurred.max() == 255 and blurred.min() == 0
    c3 = mask.max() == 255 and mask.min() == 0
    c4 = watershed.max() == 255 and watershed.min() == 0
    
    print('CONDITIONS', c1,c2,c3,c4)
    # if c1 and c2 and c3 and c4:
    if idx < split:
        io.imsave(base+'training/1st_manual/{}_manual1.gif'.format(idy), truth) 
        io.imsave(base+'training/images/{}.gif'.format(idy), blurred) # _training
        io.imsave(base+'training/mask/{}_mask.gif'.format(idy), mask) # _training
        io.imsave(base+'training/watershed/{}_watershed.gif'.format(idy), watershed)
        io.imsave(base+'training/downsampled/{}_down.gif'.format(idy), downsampled)
    else:
        io.imsave(base+'test/1st_manual/{}_manual1.gif'.format(idy), truth) 
        io.imsave(base+'test/images/{}.gif'.format(idy), blurred) # _test
        io.imsave(base+'test/mask/{}_mask.gif'.format(idy), mask) # _test
        io.imsave(base+'test/watershed/{}_watershed.gif'.format(idy), watershed)
        io.imsave(base+'test/downsampled/{}_down.gif'.format(idy), downsampled)

def generate_parameters():
    # Undulation index - sine transform in x and y position
    # Aspect ratio - varying a and b of ellipse
    # Ellipticity index - ?
    # Conicity parameter - how conelike
    # Bottleneck factor - how pinched the neck is
    ellip_sac = np.random.normal(1.2, 0.4) # np.random.uniform(0,1) # ratio for a to b
    ellip_par = np.random.normal(8.0, 1.0)  #
    und_freq_sac = np.random.uniform(1,10,2)        #)np.random.normal(mu, sigma, 2) # between 0 and x, 
    und_amp_sac = np.random.uniform(5.0, 10.0, 2)
    und_freq_par = np.random.uniform(1,3,2)         #)np.random.normal(mu, sigma, 2) # between 0 and x, 
    und_amp_par = np.random.uniform(3.0, 6.0, 2)
    vert_sac = np.random.uniform(0.9,0.95)
    sac_size = np.random.uniform(0.1,0.25)
    # und_freq_par = np.random.normal(mu, sigma)
    # und_amp_par = np.random.normal(mu, sigma)
    ellip_in = np.random.normal(1.2, 0.4)
    vert_in = np.random.uniform(0.9,0.95)
    in_size = np.random.uniform(0.1,0.25)

    ellip_complex = np.random.normal(0.5, 1.0)
    und_freq_complex = np.random.uniform(1,2,2) 
    und_amp_complex = np.random.uniform(1.0, 5.0, 2)
    vert_complex = np.random.uniform(0.9,0.95)
    complex_size = np.random.uniform(0.1,0.11)

    gaussian_noise_amount = np.random.uniform(-0.0005,0.0005)
    parameters = {}
    parameters['sac_ellipse'] = ellip_sac
    parameters['par_ellipse'] = ellip_par
    parameters['und_freq_sac'] = und_freq_sac
    parameters['und_amp_sac'] = und_amp_sac
    parameters['und_freq_par'] = und_freq_par
    parameters['und_amp_par'] = und_amp_par
    parameters['sac_vertical'] = vert_sac
    parameters['sac_size'] = sac_size

    parameters['inlet_ellipse'] = ellip_in
    parameters['inlet_vertical'] = vert_in
    parameters['inlet_size'] = in_size
    parameters['gauss_noise_amt'] = gaussian_noise_amount

    parameters['complex_ellipse'] =  ellip_complex
    parameters['complex_vertical'] = vert_complex
    parameters['complex_size'] = complex_size
    parameters['und_freq_complex'] = und_freq_complex
    parameters['und_amp_copmlex'] = und_amp_complex

    return parameters

from skimage.util import random_noise

def add_noise(image,parameters):
    # mean = 0.0
    # std = 0.2
    # image = image + np.random.normal(mean, std, img.shape)
    # print(image.max())
    # # image = image
    # img_clipped = np.clip(image, 0, 1)

    # img_clipped = random_noise(image, mode='gaussian', seed=None, clip=True, var=0.001)
    gaussian_noise_amount = parameters['gauss_noise_amt']
    # img_clipped = random_noise(image, mode='poisson', seed=None, clip=True)
    img_clipped = random_noise(image, mode='gaussian', seed=None, clip=True, mean = 0.0, var=0.001+gaussian_noise_amount)

    return img_clipped

if __name__ == '__main__':
    how_many = 60
    save_images = True

    # remove_old_figures()
    for idx, each in enumerate(range(how_many)):
        blank = create_blank(xx,yy)
        params = generate_parameters()
        img_small, img_full = generate_truth(blank,params)
        
        img_small = add_noise(img_small,params)

        img_small2large = rescale(img_small, (xx/32.0),  order=0, anti_aliasing=False) #, anti_aliasing_sigma=0.1)
        img_small2large2 = rescale(img_small, (xx/32.0), order=2, anti_aliasing=True)  #, anti_aliasing_sigma=0.1)

        img = img_full
        # img = add_noise(img)
        # img = img_noise
        blur1 = blur_image(img)
        # blur2 = blur_image(img_noise)
        # image = test_image()
        grad, watershed_seg, erode = watershed_segmentation(blur1)

        plt.figure(1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Fig 1: "Real" geometry (actually {}x{})'.format(xx,yy))
        plt.tight_layout()

        plt.figure(2)
        plt.imshow(img_small)
        plt.axis('off')
        plt.title('Fig 2: "Aquired" image (Fig 1 downsampled 16x to size 32x32')
        plt.tight_layout()

        # plt.figure(3)
        # plt.imshow(blur1)
        # plt.axis('off')
        # plt.title("Fig 3: Gaussian blur of Fig 1 (ie Dan's current method)")
        # plt.tight_layout()

        # plt.figure(4)
        # plt.imshow(img_small2large)
        # plt.axis('off')
        # plt.title("Fig 4: Upsample of Fig 2, order = 0, anti_aliasing=True")
        # plt.tight_layout()

        plt.figure(3)
        plt.imshow(img_small2large2)
        plt.axis('off')
        plt.title("Fig 5: Upsample of Fig 2, order = 2, anti_aliasing=True")
        plt.tight_layout()

        # plt.figure(6)
        # imgcomp = np.dstack((img,img_small2large))  
        # imgcomp = np.dstack((imgcomp,np.zeros((xx,yy))))  
        # plt.imshow(imgcomp)
        # plt.axis('off')
        # plt.title("Fig 6: Composite of 'Real' (Fig 1) and downsample (Fig 3)")
        # plt.tight_layout()

        # plt.figure(7)
        # img_diff = np.abs(img_small2large2-blur1)
        # plt.imshow(img_diff, vmin=0,vmax=1)
        # plt.axis('off')
        # plt.title("Fig 7: abs(Fig:5 - Fig:3), vmin=0, vmax=1")
        # plt.tight_layout()

        # plt.figure(8)
        # img_diff = np.abs(img_small2large2-blur1)
        # plt.imshow(img_diff, vmin=0,vmax=0.2)
        # plt.axis('off')
        # plt.title("Fig 8: abs(Fig:5 - Fig:3), vmin=0, vmax=0.2")
        # plt.tight_layout()

        # plt.figure(4)
        # plt.imshow(erode)
        # plt.tight_layout()
        # plt.axis('off')

        # plt.figure(5)
        # plt.imshow(watershed_seg)
        # plt.tight_layout()
        # plt.axis('off')

        # plt.figure(6)
        # plt.imshow(grad)
        # plt.tight_layout()
        # plt.axis('off')
        # show_images(img,blur,grad,watershed_seg)

        # summ = plot_overlay(img,blur,watershed_seg)

        blur_save = img_small2large2
        downsampled = img_small2large

        if save_images == True:
            # Ordering:
            # '1st_manual'  # Truth
            # 'images'      # Input image
            # 'mask'        # Subimages mask
            # 'watershed'   # Watershed of 
            # 'downsampled' # Downsampled
            save_figures(img,img_small2large2,erode,watershed_seg,downsampled,idx,0.5*how_many)
        # plt.close('all')




# from shutil import copyfile

# results_path = '/Users/danmacdonald/Google Drive/retina-unet-master/style_transfer/result/'
# destination = '/Users/danmacdonald/Google Drive/retina-unet-master/DATA/test/images/'

# styled_imgs = sorted(glob.glob(results_path + '*.png'))
# # destin = sorted(glob.glob(style_path + '*.png'))

# for file in styled_imgs:
#     name = file.split('result/')[1]
#     copyfile(file, destination + name)


#
