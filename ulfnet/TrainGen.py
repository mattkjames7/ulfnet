import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from .adjustData import adjustData

def TrainGen(image,mask,**kwargs):
	'''
	Return a training dataset generator.
	
	Inputs
	======
	image : str|float
		Input images for training.
		if str : this is the string to a path of images
		if float : this is an array of the images to use
	mask : str|float
		Output image masks for training.
		if str : this is the string to a path of image masks
		if float : this is an array of the masks to use
		
	Keyword Arguments
	=================
		
	
	'''
	
	#arguments for the ImageDataGenerator
	defgenargs = {	'rotation_range':0.2,
					'width_shift_range':0.05,
					'height_shift_range':0.05,
					'shear_range':0.05,
					'zoom_range':0.05,
					'horizontal_flip':True,
					'fill_mode':'nearest' }
	genargs = {}
	for k in list(defgenargs.keys()):				
		if k in kwargs:
			genargs[k] = kwargs[k]
		else:
			genargs[k] = defgenargs[k]
			
			
	#other flow related arguments
	image_color_mode = kwargs.get('image_color_mode','grayscale')
	mask_color_mode = kwargs.get('mask_color_mode','grayscale')
	image_save_prefix = kwargs.get('image_save_prefix','image')
	mask_save_prefix = kwargs.get('mask_save_prefix','mask')
	flag_multi_class = kwargs.get('flag_multi_class',False)
	num_class = kwargs.get('num_class',2)
	save_to_dir = kwargs.get('save_to_dir',None)
	target_size = kwargs.get('target_size',(256, 256))
	seed = kwargs.get('seed',1)
	batch_size = kwargs.get('batch_size',2)
			
	#create the generators for the images and masks
	ImageDataGen = ImageDataGenerator(**genargs)
	MaskDataGen = ImageDataGenerator(**genargs)
	
	#configure the generators based on whether we are providing arrays
	#or directories.
	if isinstance(image,str):
		#assume directory
		print('directory')
		#split the path
		train_path,cls_path = os.path.split(image)
		mask_path,mcls_path = os.path.split(mask)
		
		#do something 
		ImageGen = ImageDataGen.flow_from_directory(
			train_path,
			classes = [cls_path],
			class_mode = None,
			color_mode=image_color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=image_save_prefix,
			seed=seed)
		MaskGen = MaskDataGen.flow_from_directory(
			mask_path,
			classes=[mcls_path],
			class_mode=None,
			color_mode=mask_color_mode,
			target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=mask_save_prefix,
			seed=seed)		
	else:
		print('flow')
		print(image.shape)
		print(mask.shape)
		#hopefully our inputs are arrays!
		#do something 
		ImageGen = ImageDataGen.flow(
			image,
			#color_mode=image_color_mode,
			#target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=image_save_prefix,
			seed=seed)
		MaskGen = MaskDataGen.flow(
			mask,
			#color_mode=mask_color_mode,
			#target_size=target_size,
			batch_size=batch_size,
			save_to_dir=save_to_dir,
			save_prefix=mask_save_prefix,
			seed=seed)		
	
	#copied straight from original code...
	TrainGen = zip(ImageGen, MaskGen)
	for (img, mask) in TrainGen:
		img, mask = adjustData(img, mask, flag_multi_class, num_class)
		yield (img, mask)	
