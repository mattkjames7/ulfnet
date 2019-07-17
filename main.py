import unet_model
from data_helper import dataGenerator,load_data_Kfold, get_items, test_file_reader, plot_imgs , saveResult, plot_segm_history
#import ipdb
import numpy as np
from keras.callbacks import ModelCheckpoint
import time
from keras import backend as K 
start_time=time.time() 


BATCH_SIZE = 4
        
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                   # vertical_flip=True,
                    fill_mode='nearest')

im_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train/image'
label_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train/label'
im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'

k = 2
seed = 1

#Create folds
x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

def get_callbacks(name_weights):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    return [mcp_save]

'''
#Load model
model = unet_model.unet()
model_filename = 'segm_model_v0.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)
'''

#cv_losses=[]
#cv_acc=[]
#CV and training
for fold_number in range(k):
    x_training = get_items(x_train[fold_number])
    y_training = get_items(y_train[fold_number])
    x_valid = get_items(x_validation[fold_number])
    y_valid = get_items(y_validation[fold_number])
    print('Training fold' + str(fold_number))
    name_weights="final_model_fold" + str(fold_number) + "_weights.h5"
    callbacks=get_callbacks(name_weights = name_weights)
    generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1)
    model = unet_model.unet() 
    history=model.fit_generator(generator,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=10,verbose=1,validation_data = (x_valid,y_valid), callbacks=callbacks)
    figure = plot_segm_history(history, fold_number) 
    scores = model.evaluate(x_valid, y_valid)
    K.clear_session()
    del model
    del x_training
    del y_training
    del x_valid
    del y_valid
    #cv_losses.append(scores[0])
 
 
''' 
best_fold= cv_losses.index(min(cv_losses))
print('Best fold is ' + str(best_fold))

testGen = test_file_reader(im_test)


model.load_weights("final_model_fold" + str(best_fold) + "_weights.h5")
y_pred = model.predict(x_valid)

images= plot_imgs(org_imgs=x_valid, mask_imgs=y_valid, pred_imgs=y_pred, nm_img_to_plot=9) # COMMENTED OUT BECAUSE OF GPU ERROR - TO BE FIXED

def threshold_binarize(x, threshold=0.5):
   # boolean_array=(x>threshold)*x
    y = np.copy(x)    
    #y = (x>= threshold).astype(int)
    y[y >= threshold] = 1
    return y



time_before_predictions=time.time() 
results = model.predict_generator(testGen,2313,verbose=1)
results_thresholded = threshold_binarize(results) 
saveResult(im_test,results_thresholded)
 
print("--- %s seconds (just for predictions) ---" % (time.time() - time_before_predictions))

print("--- %s seconds (total execution time) ---" % (time.time() - start_time))

'''
