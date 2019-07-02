import unet_model
from data_helper import dataGenerator,load_data_Kfold, get_items, test_file_reader, saveResult, plot_imgs, plot_segm_history, get_x_and_y
import ipdb
import numpy as np
from keras.callbacks import ModelCheckpoint
import time

start_time=time.time() 


BATCH_SIZE = 2
        
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                   # vertical_flip=True,
                    fill_mode='nearest')

im_path = 'data/membrane/train/image'
label_path = 'data/membrane/train/label'
im_test = 'data/membrane/test'

k = 2
seed = 1

#Create folds
x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

#Load model
#model = unet_model.unet()
#model_filename = 'segm_model_v0.h5'


def get_callbacks(name_weights):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
   # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save]


'''
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)
'''

#steps_per_epoch=len(x_training)/BATCH_SIZE

cv_losses=[]
#cv_acc=[]
#CV and training
for fold_number in range(k):
    x_training = get_items(x_train[fold_number])
    y_training = get_items(y_train[fold_number])
    x_valid = get_items(x_validation[fold_number])
    y_valid = get_items(y_validation[fold_number])
    print(f'Training fold {fold_number}')
    name_weights="final_model_fold" + str(fold_number) + "_weights.h5"
    callbacks=get_callbacks(name_weights = name_weights)
    generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = 1) 
    model = unet_model.unet()
    history=model.fit_generator(generator,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=10,verbose=1,validation_data = (x_valid,y_valid), callbacks=callbacks)
    figure = plot_segm_history(history, fold_number)
    scores = model.evaluate(x_valid, y_valid)
    print(f'Data in fold {fold_number}')
    print(x_train[fold_number])
    print(x_validation[fold_number])
    print(y_train[fold_number])
    print(y_validation[fold_number])
   # print(model.evaluate(x_valid, y_valid))
   # scores = model.evaluate(x_valid, y_valid)
   # print(scores)    
    cv_losses.append(scores[0])
   # cv_acc.append(scores[1] * 100)
#Read test data and evaluate

best_fold= cv_losses.index(min(cv_losses))

testGen = test_file_reader(im_test)

#print("Loss is %.2f%% (+/- %.2f%%)" % (np.mean(cv_losses), np.std(cv_losses)))
#print("Accuracy is %.2f%% (+/- %.2f%%)" % (np.mean(cv_acc), np.std(cv_acc)))

model.load_weights("final_model_fold" + str(best_fold) + "_weights.h5")
#y_pred = model.predict(x_valid)

#images= plot_imgs(org_imgs=x_valid, mask_imgs=y_valid, pred_imgs=y_pred, nm_img_to_plot=9)


results = model.predict_generator(testGen,6,verbose=1)
saveResult("data/membrane/test",results)

calling_function=get_x_and_y("data/membrane/test")

print("--- %s seconds ---" % (time.time() - start_time))


