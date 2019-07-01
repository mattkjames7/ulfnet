import unet_model
from data_helper import dataGenerator,load_data_Kfold, get_items, test_file_reader, saveResult, plot_imgs, plot_segm_history
import numpy as np
from keras.callbacks import ModelCheckpoint
import time
import pdb
start_time=time.time() 


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





def get_callbacks(model_name):
    mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
   # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save]

#Data evaluation parameters
k = 2
SEED = 1

#Network hyperparams
#learning_rate = [0.0001, 0.001, 0.01, 0.1]
#drop_out = np.arange(0.2,0.8,0.1)
#weight_init_mode = ['he_normal','he_uniform','glorot_normal','glorot_uniform','lecun_uniform','normal','uniform']
learning_rate = [0.01,0.1]
drop_out = np.arange(0.4,0.5,0.1)
weight_init_mode = ['he_normal','he_uniform']
BATCH_SIZE = 2
cv_losses = []

# Model evaluation loop

#Create folds
x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

for fold_number in range(k):
    print(f'Training fold {fold_number}')
    x_training = get_items(x_train[fold_number])
    y_training = get_items(y_train[fold_number])
    x_valid = get_items(x_validation[fold_number])
    y_valid = get_items(y_validation[fold_number])
    
    generator = dataGenerator(BATCH_SIZE, x_training,y_training,data_gen_args,seed = SEED)     
    
    #model hyperparameter selection loop
    for learning_rate in learning_rate:
        for drop_out in drop_out:
            for weight_init_mode in weight_init_mode:
                hyper_params = list([learning_rate,drop_out,weight_init_mode])
                print(f'learning rate: {learning_rate}, drop out: {drop_out},weight mode: {weight_init_mode}')
                callbacks = get_callbacks(model_name=f'model_{learning_rate}_{drop_out}_{weight_init_mode}')
                model = unet_model.unet(learning_rate=learning_rate,drop_out=drop_out,weight_init_mode=weight_init_mode)
                history = model.fit_generator(generator,steps_per_epoch=len(x_training)/BATCH_SIZE,epochs=2,verbose=1,validation_data = (x_valid,y_valid), callbacks=callbacks)
                figure = plot_segm_history(history, fold_number)
                scores = model.evaluate(x_valid, y_valid)
                cv_losses.append(scores[0])

    
#select best model
best_fold_idx = cv_losses.index(min(cv_losses))
model.load_weights()


#Predict on test data
testGen = test_file_reader(im_test)
y_pred = model.predict(x_valid)
images= plot_imgs(org_imgs=x_valid, mask_imgs=y_valid, pred_imgs=y_pred, nm_img_to_plot=9)
results = model.predict_generator(testGen,10,verbose=1)
saveResult("data/membrane/test",results)

print("--- %s seconds ---" % (time.time() - start_time))


