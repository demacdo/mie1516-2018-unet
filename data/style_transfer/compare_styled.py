# Compare images
from skimage import io 

mask_path = '/Users/danmacdonald/Google Drive/retina-unet-master/style_transfer/con/con.png'
styled_path = '/Users/danmacdonald/Google Drive/retina-unet-master/style_transfer/result/_at_iteration_100.png'
blurred_path = '/Users/danmacdonald/Google Drive/retina-unet-master/style_transfer/0053_test.gif'

io.use_plugin('pil')

mask = io.imread(mask_path,as_gray=True)
styled = io.imread(styled_path,as_gray=True)
blurred = io.imread(blurred_path,as_gray=True)/255.0

subtract = styled-mask

plt.imshow(subtract)