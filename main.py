import unet_model 
from data import trainGenerator, testGenerator, saveResult, saveResultThresholded, threshold_binarize, plot_imgs
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
import tensorflow as tf


data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
                        
BATCH_SIZE=2
EPOCHS=5

myGene = trainGenerator(BATCH_SIZE,'/Users/maran/unet/data/membrane/train','image','label',data_gen_args,save_to_dir = None)
model = unet_model.unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=178/BATCH_SIZE,epochs=EPOCHS,callbacks=[model_checkpoint])

im_test = '/Users/maran/unet/data/membrane/test'
ground_truth_path='/Users/maran/unet/ground_truth_data'

NUM_TEST_IMAGES=12
testGene = testGenerator(im_test, NUM_TEST_IMAGES)
results = model.predict_generator(testGene,NUM_TEST_IMAGES,verbose=1)

saveResult(im_test,results)

saveResultThresholded(im_test,results, threshold=0.4)
saveResultThresholded(im_test,results, threshold=0.5)
saveResultThresholded(im_test,results, threshold=0.6)

### PREDICTION OVERLAY

## TEST IMAGE - GROUND TRUTH IMAGE - PREDICTION IMAGE - OVERLAY##
THRESHOLD= 0.6
results_thresholded = threshold_binarize(results, THRESHOLD)

from skimage import io
import os
import skimage.transform as trans

def numpy_array_generator(path, num_image):
    im_list=[]
    for i in range(num_image):
        img = io.imread(os.path.join(path,"%d.jpg"%i),as_gray = True)
        img = trans.resize(img, (256,256))
        #img = np.reshape(img,img.shape+(1,))
        #img = np.reshape(img,(1,)+img.shape)
        im_list.append(np.asarray(img))
    np_ar= np.array(im_list)
    exp_np_ar= np.expand_dims(np_ar, axis=-1)
    return exp_np_ar

test_image_np_ar=numpy_array_generator(im_test,12)
ground_truth_image_np_ar=numpy_array_generator(ground_truth_path,12)

print(test_image_np_ar.shape)
print(ground_truth_image_np_ar.shape)
print(results_thresholded.shape)
    
images = plot_imgs(org_imgs=test_image_np_ar, mask_imgs=ground_truth_image_np_ar, pred_imgs=results_thresholded, nm_img_to_plot=12)


