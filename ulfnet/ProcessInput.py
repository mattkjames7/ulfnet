import numpy as np


def ProcessInput(image,mask,**kwargs):
	'''
	Using the ImageDataGenerator from keras, we manipulate input images
	either from file or directly from arrays.
	
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
