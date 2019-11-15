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

BATCH_SIZE = 2
myGene = trainGenerator(BATCH_SIZE, '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train',
                        'image', 'label', data_gen_args, save_to_dir=None)

model = unet_model.unet()
model_checkpoint = ModelCheckpoint(
    'unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
history = model.fit_generator(
    myGene, steps_per_epoch=178/BATCH_SIZE, epochs=10, callbacks=[model_checkpoint])
figure = plot_segm_history(history)

im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'

# Since 178 images out of 2313 were used for taining, the test dataset actually only includes 2135 days. For simplicity, predictions were generated for all 2313 days.

NUM_TEST_IMAGES = 2313

testGene = testGenerator(im_test, NUM_TEST_IMAGES)
results = model.predict_generator(testGene, NUM_TEST_IMAGES, verbose=1)

saveResult(im_test, results)
saveResultThresholded(im_test, results, threshold=0.5)
