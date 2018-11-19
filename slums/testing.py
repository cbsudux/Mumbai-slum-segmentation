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
from sklearn.metrics import jaccard_similarity_score

import argparse


def get_ax(rows=1, cols=1, size=16):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.
	
	Adjust the size attribute to control how big to render images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax

def getLargestCC(segmentation):
	labels = label(segmentation) #Gives different integer value for each connected region
	#largest region is background. Ignore that and get second largest.
	if len(np.bincount(labels.flat))==1:
		return labels

	largest_val = np.argmax(np.bincount(labels.flat)[1:]) + 1 #+1 as we ignore bg
	#print(np.bincount(labels.flat)[1:],' Largest ',largest_val)
	return labels==largest_val		

def load_model():
	with tf.device(DEVICE):
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)	
	weights_path = SLUM_WEIGHTS_PATH

	# Load weights
	print("Loading weights ", weights_path)
	model.load_weights(weights_path, by_name=True)		
	return model
		

def compute_batch_ap(dataset, image_ids, verbose=1):
	"""
	# Load validation dataset if you need to use this function.
	dataset = slum.slumDataset()
	dataset.load_slum(folder_path,fol)

	"""

	APs = []
	IOUs = []

	for image_id in image_ids:
		# Load image
		image, image_meta, gt_class_id, gt_bbox, gt_mask =\
			modellib.load_image_gt(dataset, config,
								   image_id, use_mini_mask=False)

		# Run object detection
		results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
		# Compute AP over range 0.5 to 0.95
		r = results[0]

		#merge_masks.
		gt_merge_mask = np.zeros((gt_mask.shape[:2]))
		for i in range(gt_mask.shape[2]):
			gt_merge_mask = np.logical_or(gt_merge_mask,gt_mask[:,:,i])

		pred_merge_mask = np.zeros((r['masks'].shape[:2]))
		for i in range(r['masks'].shape[2]):
			pred_merge_mask = np.logical_or(pred_merge_mask,r['masks'][:,:,i])
		
		
		pred_merge_mask = np.expand_dims(pred_merge_mask,2)
		#print(pred_merge_mask.shape)
		pred_merge_mask,wind,scale,pad,crop = utils.resize_image(pred_merge_mask,1024,1024)
		#print(pred_merge_mask.shape,gt_merge_mask.shape)
		
		iou = jaccard_similarity_score(np.squeeze(pred_merge_mask),gt_merge_mask)

		#mAP at 50
		print("mAP at 50")
		ap = utils.compute_ap_range(
			gt_bbox, gt_class_id, gt_mask,
			r['rois'], r['class_ids'], r['scores'], r['masks'],np.arange(0.5,1.0),verbose=0)
		
		#Make sure ap doesnt go above 1 !
		if ap>1.0:
			ap = 1.0

		APs.append(ap)
		IOUs.append(iou)

		if verbose:
			info = dataset.image_info[image_id]
			meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
			print("{:3} {}   AP: {:.2f} Image_id: {}, IOU: {}".format(
				meta["image_id"][0], meta["original_image_shape"][0], ap,image_id,iou))
	return APs,IOUs


def test_on_folder(model,folder_path,save_path='test_outputs/'):
	
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	files = glob.glob(folder_path+'/*.jpg')

	for i in range(len(files)):
		image_id = i
		image = skimage.io.imread(files[image_id])
		results = model.detect(image[np.newaxis],verbose=0)
		results = results[0]
		class_names = ['slum']*(len(results['class_ids'])+1)
		mask = results['masks']

		file_to_save = save_path + '/pred_'+str(image_id) + '.jpg'

		visualize.save_instances(image, results['rois'], results['masks'], results['class_ids'], 
								 class_names,file_to_save,results['scores'], ax=None,
								 show_bbox=False, show_mask=True,
								 title="Predictions "+str(image_id))

		#Uncomment to visualize using matpltolib.
		"""
		visualize.display_instances(image, resukts['rois'], results['masks'], results['class_ids'], 
								 class_names, results['scores'], ax=get_ax(0),
								 show_bbox=False, show_mask=True,
								 title="Predictions "+str(image_id))
		"""


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--weights_path",type=str,required=True)
	args = parser.parse_args()

	SLUM_WEIGHTS_PATH = args.weights_path


	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	config = slum.slumConfig()
	

	class InferenceConfig(config.__class__):
		# Run detection on one image at a time
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()

	DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

	# Inspect the model in training or inference modes
	# values: 'inference' or 'training'
	# TODO: code for 'training' test mode not ready yet
	TEST_MODE = "inference"

	#Use to run over test_data
	model = load_model()
	test_on_folder(model,'test_images/')