import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import scipy
import imageio, imutils
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, decimate, resample

def downsample_naive(img, downsample_factor):
	"""
	Naively downsamples image without LPF.
	"""
	new_img = img.copy()
	new_img = new_img[::downsample_factor]
	new_img = new_img[:,::downsample_factor]
	return new_img

def locate_block(img, mask=None, downsample_factor=4, correlated=None):
	if len(img.shape) == 3:
		img = img[:,:,0]
	if mask is None:
		mask = np.load("mask.npy")
	# TODO: downsample mask too
	nonzero = (img > 0).astype(float)
	nonzero = downsample_naive(nonzero, downsample_factor)
	downsampled_mask = downsample_naive(mask, downsample_factor)
	if correlated is None:
		correlated = correlate2d(nonzero, downsampled_mask, mode='same')
	best = np.array(np.unravel_index(correlated.argmax(), nonzero.shape)) * downsample_factor
	best[0] -= mask.shape[0]//2
	best[1] -= mask.shape[1]//2

	new_img = np.zeros_like(img)
	new_img[best[0]:best[0] + mask.shape[0],best[1]:best[1] + mask.shape[1]] = mask
	return (best, mask, correlated.max(), correlated)

def render_mask(img, mask, start):
	new_img = np.zeros_like(img)
	plt.imshow(mask); plt.show()
	new_img[start[0]:start[0] + mask.shape[0],start[1]:start[1] + mask.shape[1]] = mask
	plt.imshow(img); plt.show()
	plt.imshow(new_img); plt.show()
	plt.imshow(new_img + img); plt.show()
	return np.multiply(new_img, img)

def get_masked_image(img, mask, start):
	new_img = np.zeros_like(img)
	new_img[start[0]:start[0] + mask.shape[0],start[1]:start[1] + mask.shape[1]] = mask
	return np.multiply(new_img, img)

def zero_correlated(correlated, start, mask, downsample_factor=4):
	start_downsampled = start // downsample_factor
	buffer_scale = 1.2
	downsampled_mask = downsample_naive(mask, downsample_factor)
	correlated[start_downsampled[0]:start_downsampled[0] + int(downsampled_mask.shape[0] * buffer_scale),
			start_downsampled[1]:start_downsampled[1] + int(downsampled_mask.shape[1] * buffer_scale)] = 0
	return correlated

def rotate_mask(angle):
	mask = np.load("mask.npy")
	rotated = imutils.rotate_bound(mask, angle)
	rotated[rotated>0] = 1
	return rotated

def find_masks(img, num_triangles):
	masks = [rotate_mask(i) for i in np.r_[0:120:4]]
	ret = []
	angles = []
	for mask in masks:
		ret.append(locate_block(img, mask))
	best_value = np.argmax([r[2] for r in ret])
	angle = np.r_[0:120:4][best_value]
	angles.append(angle)
	mask_im = get_masked_image(img, ret[best_value][1], ret[best_value][0])
	# print(ret[best_value][0])
	# plt.imshow(mask_im); plt.show()
	np.save("mask%d.npy"%0, mask_im)

	for j in range(1, num_triangles):
		corr = [zero_correlated(r[3], ret[best_value][0], ret[best_value][1]) for r in ret]
		ret = []
		for i, mask in enumerate(masks):
			ret.append(locate_block(img, mask, correlated=corr[i]))
		best_value = np.argmax([r[2] for r in ret])
		angle = np.r_[0:120:4][best_value]
		angles.append(angle)
		mask_im = get_masked_image(img, ret[best_value][1], ret[best_value][0])
		# print(ret[best_value][0])
		np.save("mask%d.npy"%j, mask_im)
	corr = [zero_correlated(r[3], ret[best_value][0], ret[best_value][1]) for r in ret]
	np.save("angles.npy", np.array(angles))

if __name__ == '__main__':
	file_name = "../img/block_mask.png"
	mask = imageio.imread(file_name)


	rotated = imutils.rotate_bound(mask, 30)

	cv2.imshow("", rotated)
	cv2.waitKey(0)
	# find_masks(img, 6)
    #
	# out = np.zeros_like(img)
    #
	# for i in range(12):
	# 	data = np.load("mask%d.npy"%i)
	# 	out += data
	# 	plt.imshow(data); plt.show()
    #
	# plt.imshow(out); plt.show()
	# plt.imshow(img); plt.show()
	# plt.imshow(np.abs(img - out)); plt.show()
	# print(np.load("angles.npy"))