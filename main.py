import unet_model 
from data import test_file_reader, trainGenerator, saveResult, plot_segm_history
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
                    
BATCH_SIZE=2
myGene = trainGenerator(BATCH_SIZE,'/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet_model.unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])
figure = plot_segm_history(history)


im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'

testGene = test_file_reader(im_test)
results = model.predict_generator(testGene,36,verbose=1)

'''
def threshold_binarize(x, threshold=0.5):
   # boolean_array=(x>threshold)*x
    y = np.copy(x)    
    #y = (x>= threshold).astype(int)
    y[y >= threshold] = 1
    return y

results_thresholded = threshold_binarize(results) 
'''

saveResult(im_test,results)
