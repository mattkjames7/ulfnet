import time
start_time = time.time()

import unet_model 
from data import trainGenerator, testGenerator, saveResult, saveResultThresholded, threshold_binarize, plot_imgs, plot_segm_history
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras import backend as K
import tensorflow as tf
import time

end_imports= time.time()

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#with K.tf.device('/gpu:3'):

#start_time= time.time()
print(" %s seconds for imports---" % (end_imports - start_time))

start_time_initialization = time.time()




data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
                        
BATCH_SIZE=2
myGene = trainGenerator(BATCH_SIZE,'/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train','image','label',data_gen_args,save_to_dir = None)
im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'
NUM_TEST_IMAGES=2135    
model = unet_model.unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

end_init= time.time()

print(" %s seconds for initialization---" % (end_init - start_time_initialization))

start_training= time.time()

history = model.fit_generator(myGene,steps_per_epoch=178/BATCH_SIZE,epochs=2,callbacks=[model_checkpoint])

end_training= time.time()

print(" %s seconds for training---" % (end_training - start_training))
#print(" %s seconds total execution time---" % (end_training - start_time))


start_plotting = time.time()

figure = plot_segm_history(history)

end_plotting = time.time()
#im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'
#im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data_large/test'

print(" %s seconds for plotting---" % (end_plotting - start_plotting))

#NUM_TEST_IMAGES=2313

#NUM_TEST_IMAGES=2135

start_predicting= time.time()

testGene = testGenerator(im_test, NUM_TEST_IMAGES)
results = model.predict_generator(testGene,NUM_TEST_IMAGES,verbose=1)


#THRESHOLD=0.5

saveResult(im_test,results)
#saveResultThresholded(im_test,results, threshold=0.4)
saveResultThresholded(im_test,results, threshold=0.5)
#saveResultThresholded(im_test,results, threshold=0.6)

end_predicting = time.time()

print(" %s seconds for predicting---" % (end_predicting - start_predicting))
print(" %s seconds total execution time---" % (end_predicting - start_time))


'''


'''

### PREDICTION OVERLAY

## TEST IMAGE - GROUND TRUTH IMAGE - PREDICTION IMAGE - OVERLAY##
'''

ground_truth_path='/lustre/home/d167/s1137563/Paolo_repository/unet/ground_truth_data'
#ground_truth_data=testGenerator(ground_truth_path, NUM_TEST_IMAGES)



results_thresholded = threshold_binarize(results, 0.5)

'''



#my_array = np.empty(12)
#for i, el in enumerate(testGene): my_array[i] = el
#print(my_array)
#print(my_array.shape)
#next(testGene)

'''
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

#print(test_image_np_ar.shape)
#print(ground_truth_image_np_ar.shape)
#print(results_thresholded.shape)

'''

#x = np.stack(testGene)
#print(x.shape)

#images = plot_imgs(org_imgs=test_image_np_ar, mask_imgs=ground_truth_image_np_ar, pred_imgs=results_thresholded, nm_img_to_plot=12)



'''
def threshold_binarize(x, threshold=0.5):
   # boolean_array=(x>threshold)*x
    y = np.copy(x)    
    #y = (x>= threshold).astype(int)
    y[y >= threshold] = 1
    return y///

results_thresholded = threshold_binarize(results) 
'''
