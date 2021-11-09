import os
import shutil
import random

path_list = [
	'/home/xqwang/projects/saliency/semi-sod/datasets/RGBD135',
	'/home/xqwang/projects/saliency/semi-sod/datasets/SIP',
	'/home/xqwang/projects/saliency/semi-sod/datasets/STEREO',
	'/home/xqwang/projects/saliency/semi-sod/datasets/LFSD'
]

ns = [ 10, 10, 5, 5]

dest = '/home/xqwang/projects/saliency/semi-sod/datasets/VAL'
random.seed(65535)

def copy_n(l, src, n):
	dest_images = os.path.join(dest, 'test_images')
	dest_depth = os.path.join(dest, 'test_depth')
	dest_masks = os.path.join(dest, 'test_masks')

	src_images_dir = os.path.join(src, 'test_images') # jpg
	src_depth_dir = os.path.join(src, 'test_depth') # png
	src_masks_dir = os.path.join(src, 'test_masks') # png

	random.shuffle(l)
	for i in range(n):
		src_image_name = os.path.join(src_images_dir, l[i] + '.jpg')
		src_depth_name = os.path.join(src_depth_dir, l[i] + '.png')
		src_mask_name = os.path.join(src_masks_dir, l[i] + '.png')
		if os.path.exists(src_image_name) and os.path.exists(src_depth_name) and os.path.exists(src_mask_name):
			shutil.copy(src_image_name, dest_images)
			shutil.copy(src_depth_name, dest_depth)
			shutil.copy(src_mask_name, dest_masks)
		else:
			print('{} has error about extension name'.format(src))

def build_list(d):
	l = os.listdir(os.path.join(d, 'test_images')) # relative path
	l = list(map(lambda f: os.path.splitext(f)[0], l))
	return l

def main():
	N = len(path_list)
	for i in range(N):
		l = build_list(path_list[i])
		copy_n(l, path_list[i], ns[i])

if __name__ == '__main__':
	main()