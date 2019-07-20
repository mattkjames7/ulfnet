import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys
from PIL import Image
from skimage import io
import skimage.transform as trans

from keras_unet.utils import load_data_Kfold, get_items

from keras_unet.utils import get_augmented
from keras.optimizers import Adam, SGD
from keras_unet.models import custom_unet
from keras_unet.metrics import iou, iou_thresholded

from keras.callbacks import ModelCheckpoint
from keras_unet.models import custom_unet
#from keras_unet.utils import plot_segm_history

import time

start_time = time.time()


BATCH_SIZE = 2



label_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/input/train/label_test'
im_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/input/train/im_test'
k = 2

#folds, x_train, y_train = load_data_Kfold(org_path,mask_path,k)


def get_callbacks(name_weights):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
   # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save]
'''
model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)
'''




#ADD IOU METRIC?

data_gen_args = dict(
    rotation_range=15.,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=50,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant'
)





x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

cv_losses=[]
for fold_number in range(k):
    x_training = get_items(x_train[fold_number])
    y_training = get_items(y_train[fold_number])
    x_valid = get_items(x_validation[fold_number])
    y_valid = get_items(y_validation[fold_number])
    print('Training fold' + str(fold_number))
    #generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
    train_gen = get_augmented(x_training, y_training, batch_size=2, seed=1)
    name_weights="final_model_fold" + str(fold_number) + "_weights.h5"
    callbacks=get_callbacks(name_weights = name_weights)
    model = custom_unet(
    (512,512,1),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')
    model.compile(
    optimizer=Adam(lr=0.0001), 
    loss='binary_crossentropy',
    metrics=[iou, iou_thresholded, 'accuracy'])
    history = model.fit_generator(train_gen,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=10,verbose=1,validation_data = (x_valid,y_valid),callbacks=callbacks)
    #figure = plot_segm_history(history, fold_number)
    #scores = model.evaluate(x_valid, y_valid)
    #cv_losses.append(scores[0])
    #history = model.fit_generator(train_gen,steps_per_epoch=2,epochs=3,verbose=1,validation_data = (x_valid,y_valid),callbacks=[callback_checkpoint])
    #scores = model.evaluate(x_valid, y_valid)
    #print(scores)    
    #cv_losses.append(scores[0]*100)
    #cv_acc.append(scores[1] * 100)

#print("Loss is %.2f%% (+/- %.2f%%)" % (np.mean(cv_losses), np.std(cv_losses)))
#print("Accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cv_acc), np.std(cv_acc)))

#print(history.history['val_loss'])

#from keras_unet.metrics import threshold_binarize
#import tensorflow as tf
#from keras_unet.utils import plot_segm_history


#figure = plot_segm_history(history)

'''
best_fold= cv_losses.index(min(cv_losses))
print('Best fold is ' +str(best_fold))

model.load_weights("final_model_fold" + str(best_fold) + "_weights.h5")
y_pred = model.predict(x_valid)
#print(type(y_pred))i
#print(y_pred)

#data_tf = tf.convert_to_tensor(y_pred,np.float32)
#print(type(data_tf))

#y_pred_thresholded = threshold_binarize(data_tf, threshold=0.7)
#print(type(y_pred_thresholded))

#y_pred_thresholded_np=tf.Session().run(y_pred_thresholded)
#print(type(y_pred_thresholded_np))
#print(y_pred_thresholded_np)
#y_pred_thresholded_np =y_pred_thresholded.numpy()
#print(type(y_pred_thresholded_np)) 
#print('hello' seconds ---" % (time.time() - start_time))

from keras_unet.utils import plot_imgs, test_file_reader, saveResult  

images= plot_imgs(org_imgs=x_valid, mask_imgs=y_valid, pred_imgs=y_pred, nm_img_to_plot=9)
print(x_valid)
print(x_validation[fold_number])
#images= plot_imgs(org_imgs=x_valid, mask_imgs=y_pred, pred_imgs=y_pred_thresholded_np, nm_img_to_plot=9)

testGen = test_file_reader('input/test')
results = model.predict_generator(testGen,10,verbose=1)
saveResult("input/test",results)

'''

print("--- %s seconds ---" % (time.time() - start_time))
