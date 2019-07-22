import unet_model 
from data import trainGenerator, testGenerator, saveResult, saveResultThresholded
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
history = model.fit_generator(myGene,steps_per_epoch=178/BATCH_SIZE,epochs=10,callbacks=[model_checkpoint])
#figure = plot_segm_history(history)


im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'
#im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data_large/test'


NUM_TEST_IMAGES=2313
#NUM_TEST_IMAGES=10



testGene = testGenerator(im_test, NUM_TEST_IMAGES)
results = model.predict_generator(testGene,NUM_TEST_IMAGES,verbose=1)


#THRESHOLD=0.5

saveResult(im_test,results)
saveResultThresholded(im_test,results, threshold=0.4)
saveResultThresholded(im_test,results, threshold=0.5)
saveResultThresholded(im_test,results, threshold=0.6)


### PREDICTION OVERLAY

## TEST IMAGE - GROUND TRUTH IMAGE - PREDICTION IMAGE - OVERLAY##

images= plot_imgs(org_imgs=testGene, mask_imgs=y_pred, pred_imgs=y_pred_thresholded, nm_img_to_plot=9)


'''
def threshold_binarize(x, threshold=0.5):
   # boolean_array=(x>threshold)*x
    y = np.copy(x)    
    #y = (x>= threshold).astype(int)
    y[y >= threshold] = 1
    return y

results_thresholded = threshold_binarize(results) 
'''
