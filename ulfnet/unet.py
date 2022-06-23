import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class unet(object):
	def __init__(self,**kwargs):
		'''
		Initializer for the unet object.
		
		'''
		#settings for this network
		self.InputSize = kwargs.get('InputSize',(256,256,1))
		self.Loss = kwargs.get('Loss','binary_crossentropy')
		self.HiddenAF = kwargs.get('HiddenAF','relu')
		self.OutputAF = kwargs.get('OutputAF','sigmoid')
		self.Optimizer = kwargs.get('Optimizer','Adam')
		self.OptimizerOpts = kwargs.get('OptimizerOpts',{'learning_rate':1e-4})
		
		#set up activation functions
		if self.HiddenAF == 'LeakyReLU':
			self._HidAF = layers.LeakyReLU()
		else:
			self._HidAF = self.HiddenAF
		if self.OutputAF == 'LeakyReLU':
			self._OutAF = layers.LeakyReLU()
		else:
			self._OutAF = self.OutputAF

		#and the optimizer
		try:
			self._Opt = getattr(keras.optimizers,self.Optimizer)(**self.OptimizerOpts)
		except:
			print('Something went wrong while trying to initialize the {:s} optimizer.'.format(self.Optimizer))
			print('Using the default (Adam)')
			self.Optimizer = 'Adam'
			self.OptimizerOpts = {'learning_rate':1e-4}
			self._Opt = keras.optimizers.Adam(learning_rate=1e-4)
		
		#empty variables for tracing training progress
		self.Jt = np.array([],dtype='float64')
		self.Jc = np.array([],dtype='float64')
		self.At = np.array([],dtype='float64')
		self.Ac = np.array([],dtype='float64')
		self.IoUt = np.array([],dtype='float64')
		self.IoUc = np.array([],dtype='float64')
		self.hist = []
		
		#init the model
		self._CreateModel()
		
	def _CreateModel(self):
		'''
		This bit of code is taken directly from the original repo from
		which this was forked. It initializes the model object using 
		keras.
		
		
		'''
			
		
		inputs = Input(self.InputSize)
		conv1 = Conv2D(64, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		conv2 = Conv2D(128, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		conv3 = Conv2D(256, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		conv4 = Conv2D(512, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation=self._HidAF, padding='same',
					 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
		merge6 = concatenate([drop4, up6], axis=3)
		conv6 = Conv2D(512, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv6)

		up7 = Conv2D(256, 2, activation=self._HidAF, padding='same',
					 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
		merge7 = concatenate([conv3, up7], axis=3)
		conv7 = Conv2D(256, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv7)

		up8 = Conv2D(128, 2, activation=self._HidAF, padding='same',
					 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
		merge8 = concatenate([conv2, up8], axis=3)
		conv8 = Conv2D(128, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv8)

		up9 = Conv2D(64, 2, activation=self._HidAF, padding='same',
					 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
		merge9 = concatenate([conv1, up9], axis=3)
		conv9 = Conv2D(64, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation=self._HidAF, padding='same',
					   kernel_initializer='he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation=self.OutAF)(conv9)

		self.model = Model(inputs=inputs, outputs=conv10)
		self.model.compile(optimizer=self._Opt,
				loss=self.Loss, metrics=[self.Loss,'accuracy', iou])	
