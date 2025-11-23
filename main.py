
import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
out_img_hsv = colorize_hsv(img)
out_img_pseudocolor_jet = colorize_pseudocolor(img, colormap_mode='jet')
out_img_pseudocolor_viridis = colorize_pseudocolor(img, colormap_mode='viridis')



# Save all outputs
# plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
# plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)
# plt.imsave('%s_hsv.png'%opt.save_prefix, out_img_hsv)
# plt.imsave('%s_pseudocolor_jet.png'%opt.save_prefix, out_img_pseudocolor_jet)
# plt.imsave('%s_pseudocolor_viridis.png'%opt.save_prefix, out_img_pseudocolor_viridis)

# Display all colorization results in a grid
plt.figure(figsize=(18, 12))
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(img_bw)
plt.title('Input (Grayscale)')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(out_img_eccv16)
plt.title('ECCV 16')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(out_img_siggraph17)
plt.title('SIGGRAPH 17')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(out_img_hsv)
plt.title('HSV-based')
plt.axis('off')


plt.subplot(3, 3, 6)
plt.imshow(out_img_pseudocolor_jet)
plt.title('Pseudocolor (Jet)')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(out_img_pseudocolor_viridis)
plt.title('Pseudocolor (Viridis)')
plt.axis('off')



plt.tight_layout()
plt.show()
