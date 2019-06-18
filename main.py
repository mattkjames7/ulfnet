import numpy as np
import matplotlib.pyplot as plt

import glob
import os
import sys
from PIL import Image
from skimage import io
import skimage.transform as trans

from keras_unet.utils import load_data_Kfold, get_items

mask_path = "input/train/label_test"
org_path = "input/train/im_test"
k = 2

folds, x_train, y_train = load_data_Kfold(org_path,mask_path,k)

from keras.callbacks import ModelCheckpoint
from keras_unet.models import custom_unet
model = custom_unet(
    (512,512,1),
    use_batch_norm=False,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid')

model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)

from keras_unet.utils import get_augmented
from keras.optimizers import Adam, SGD
from keras_unet.models import custom_unet
BATCH_SIZE = 2
for fold_number, (train_idx,val_idx) in enumerate(folds):
    print(f'Training fold {fold_number}')
    x_training = get_items(x_train[train_idx])
    y_training = get_items(y_train[train_idx])
    x_valid = get_items(x_train[val_idx])
    y_valid = get_items(y_train[val_idx])
    input_shape = x_training[0].shape
    
    model.compile(
    optimizer=Adam(lr=0.0001), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    #metrics=[iou, iou_thresholded]
    metrics = ['accuracy'])

    #ADD IOU METRIC?

    train_gen = get_augmented(
    x_training, y_training, batch_size=2,
    data_gen_args = dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))
    #generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
    history = model.fit_generator(train_gen,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=1,verbose=1,validation_data = (x_valid,y_valid),callbacks=[callback_checkpoint])
    #history = model.fit_generator(train_gen,steps_per_epoch=2,epochs=3,verbose=1,validation_data = (x_valid,y_valid),callbacks=[callback_checkpoint])


print(history.history['val_loss'])


from keras_unet.utils import plot_segm_history

#figure = plot_segm_history(history)

model.load_weights(model_filename)
y_pred = model.predict(x_valid)

from keras_unet.utils import plot_imgs

images= plot_imgs(org_imgs=x_valid, mask_imgs=y_valid, pred_imgs=y_pred, nm_img_to_plot=9)

#ADD CODE TO PREDICT ON TEST IMAGES
