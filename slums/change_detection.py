import os
import sys,glob
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from slums import slum

import skimage.draw
from skimage import measure
from shapely.geometry.polygon import Polygon
from skimage.measure import label   


import argparse

#get largest conncected component in each mask
def getLargestCC(segmentation):
	labels = label(segmentation) #Gives different integer value for each connected region
	#largest region is background. Ignore that and get second largest.
	largest_val = np.argmax(np.bincount(labels.flat)[1:]) + 1 #+1 as we ignore bg
	#print(np.bincount(labels.flat)[1:],' Largest ',largest_val)
	return labels==largest_val

def merge_masks(masks):
	print('No of masks: ',masks.shape[2])
	if masks.shape[2] <=1:
		return masks
	
	merged_mask_list = []
	not_required = [] #list of indices not required as masks are merged
	for i in range(masks.shape[2]):
		m = masks[:,:,i]
		m = getLargestCC(m)
		m = np.expand_dims(m,axis=2)
		
		max_iou = -1
		max_mask = -1
		max_iou_index = -1

		#Calculate max_iou with other masks.
		for j in range(masks.shape[2]):
			#Same mask gives 1.0 !
			if j!=i:
				n = masks[:,:,j]
				n = np.expand_dims(n,axis=2)
				intersection = np.logical_and(m,n)
				union = np.logical_or(m,n)
				iou_score = np.sum(intersection) / np.sum(union)   
				#print(np.sum(intersection),np.sum(union))
				#print(iou_score)
				if iou_score > max_iou:
					max_iou = iou_score
					max_mask = n
					max_iou_index = j
		
		#Need to merge if greater than 0.2
		if max_iou > 0.15:
			area_m = measure.regionprops(m[:,:,0].astype(np.uint8))
			area_m = [prop.area for prop in area_m][0]
			#print(area_m,i)
			area_max_mask = measure.regionprops(max_mask[:,:,0].astype(np.uint8))
			area_max_mask = [prop.area for prop in area_max_mask][0]
			#print(area_max_mask,max_iou_index)
			
			#print(area_m/(area_m + area_max_mask))
			#print(area_max_mask/(area_m + area_max_mask))
			
			if area_m >= area_max_mask:
				merged_mask_list.append(m)
				not_required.append(max_iou_index) 
			else:
				merged_mask_list.append(max_mask)
				not_required.append(i)
		
		elif i not in not_required:
			merged_mask_list.append(m)
		
		#print('Matches: ',max_iou,i,max_iou_index)
		#print(not_required,len(merged_mask_list))
		
	merged_mask_list = np.array(merged_mask_list)
	merged_mask_list = np.squeeze(merged_mask_list)
	merged_mask_list = np.transpose(merged_mask_list,(1,2,0))
	
	return merged_mask_list


def load_model():
	with tf.device('/gpu:0'):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)	
	weights_path = SLUM_WEIGHTS_PATH

	# Load weights
	print("Loading weights ", weights_path)
	model.load_weights(weights_path, by_name=True)		
	return model


def get_area(mask):
	area = measure.regionprops(mask.astype(np.uint8))	
	area = [prop.area for prop in area][0]
	return area

def cal_diff(mask_1,mask_2,files,image_1,image_2,results_1,results_2):
	len_1 = mask_1.shape[2]
	len_2 = mask_2.shape[2]

	#Number of detections might be unequal
	#combine mask channels.
	m1 = np.zeros((mask_1.shape[:2]))
	for i in range(len_1):
		m1 = np.logical_or(m1,mask_1[:,:,i])

	m2 = np.zeros((mask_2.shape[:2]))
	for i in range(len_2):
		m2 = np.logical_or(m2,mask_2[:,:,i])

	
	#Calculate total area covered by mask_1
	mask_1_area = get_area(m1)
	mask_2_area = get_area(m2)

	m1 = m1.astype(np.uint8)	
	m2 = m2.astype(np.uint8)	

	print(m1.shape)
	print(m2.shape)

	diff = cv2.absdiff(m1,m2)
	diff_area = get_area(diff)

	print("M1 area :",mask_1_area)
	print("M2 area :",mask_2_area)
	print("Diff in area :",diff_area)

	max_area = max(mask_1_area,mask_2_area)

	d = diff_area/max_area
	if mask_1_area > mask_2_area:
		print(files[0],' greater area')
	else:
		print(files[1],' greater area')

	print('Change ',d*100,'%')

	return m1,m2,diff

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--weights_path",type=str,required=True)
	args = parser.parse_args()

	SLUM_WEIGHTS_PATH = args.weights_path


	config = slum.slumConfig()
	class InferenceConfig(config.__class__):
		# Run detection on one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1
	config = InferenceConfig()
	config.display()


	MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	model = load_model()


	files = glob.glob('change_det/*.jpg')

	image_1 = skimage.io.imread(files[0])
	image_2 = skimage.io.imread(files[1])

	results_1 = model.detect(image_1[np.newaxis],verbose=0)
	results_2 = model.detect(image_2[np.newaxis],verbose=0)

	mask_1 = results_1[0]['masks']
	mask_2 = results_2[0]['masks']

	mask_1,mask_2,diff =cal_diff(mask_1,mask_2,files,image_1,image_2,results_1,results_2)


	r = results_2[0]
	r['masks'] = merge_masks(r['masks'])
	class_names = ['slum']*(len(r['class_ids'])+1)

	visualize.display_instances(image_2, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], ax=None,show_bbox=False,show_mask=True,
								title=files[0])


	r = results_1[0]
	r['masks'] = merge_masks(r['masks'])
	class_names = ['slum']*(len(r['class_ids'])+1)

	visualize.display_instances(image_1, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], ax=None,show_bbox=False,show_mask=True,
								title=files[1])



	print(files,' FILES')

	plt.imshow(mask_1)
	plt.axis('off')
	plt.savefig('change_det/mask_1.png',bbox_inches='tight')
	#plt.show()

	plt.imshow(mask_2)
	plt.axis('off')
	plt.savefig('change_det/mask_2.png',bbox_inches='tight')
	#plt.show()

	plt.imshow(diff)
	plt.axis('off')
	plt.savefig('change_det/change.png',bbox_inches='tight')
	#plt.show()