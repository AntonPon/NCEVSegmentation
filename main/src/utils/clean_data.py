import os
import cv2

'''
get_dirs = '../../../data/anpon/cityscapes/leftImg8bit/train'

for folder in os.listdir(get_dirs):
	dir_folder = os.path.join(get_dirs, folder)
	for image in os.listdir(dir_folder):
		if image[-3:] == 'png':
			img = cv2.imread(os.path.join(dir_folder, image))
			if img is None:
				print(image)
'''


if __name__ == '__main__':
    data_dir = '../../../../../../data/anpon/cityscapes/leftImg8bit/val'
    data_full_dir = '../../../../../../data/anpon/cityscapes2/leftImg8bit_sequence/val'

    total_elements = 6 # frame number has format 000xxx
    for folder in os.listdir(data_dir):
        dir_folder = os.path.join(data_dir, folder)
        for image in os.listdir(dir_folder):
            parts = image.split('_')
            for i in range(1, 11):
                img_to_del = str(int(parts[2]) + 1)
                zeros = (total_elements - len(img_to_del)) * '0'
                img_to_del = '{}{}'.format(zeros, img_to_del)
                parts[2] = img_to_del
                path_to_del = os.path.join(data_full_dir, folder, '_'.join(parts))
                os.remove(path_to_del)