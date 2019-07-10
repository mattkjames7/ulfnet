import unet_model
from data_helper import dataGenerator, find_parameters, load_data_Kfold, get_items # test_file_reader, plot_imgs, plot_segm_history
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import pandas as pd
import os
import tensorflow as tf

start_time=time.time() 


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                   # vertical_flip=True,
                    fill_mode='nearest')

#im_path = 'data/membrane/train/image'
#label_path = 'data/membrane/train/label'
#im_test = 'data/membrane/test'

im_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train/image'
label_path = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/train/label'
im_test = '/lustre/home/d167/s1137563/Paolo_repository/unet/data/membrane/test'



'''
def get_callbacks(model_name):
    mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
   # reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save]
'''


#Data evaluation parameters
k = 2
SEED = 1

#Network hyperparameters
learning_rate = [0.0001, 0.001, 0.01, 0.1]
drop_out = np.arange(0.2,0.8,0.1)
weight_init_mode = ['he_normal','he_uniform','glorot_normal','glorot_uniform','lecun_uniform','normal','uniform']
batch_size = [2,4,16,32,64]
optimizer=['SGD', 'Adam']


###CHECK OUTPUT FILE FROM PREVIOUS RUNS TO SEE IF YOU SHOULD CHANGED THESE CALLBACKS

early_stop = EarlyStopping(monitor='val_loss', patience=1, verbose=1, min_delta=0.1)
#reduce_lr = ReduceLROnPlateau(factor=0.3,patience=2, min_lr=0.000001, verbose=1)
callbacks=[early_stop]



#Create folds
x_train,x_validation,y_train,y_validation = load_data_Kfold(im_path,label_path,k)

#Create permutations
cv_losses = []
permutations = find_parameters(learning_rate,drop_out, weight_init_mode, batch_size, optimizer)

#Model hyperparameter selection loop
for idx_perms in range(len(permutations)):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print('Permutation number: ' +str(idx_perms)) 
    
    cv_losses_temp = []
    curr_lr = permutations[idx_perms,0]
    curr_drop_out =  permutations[idx_perms,1]
    curr_init = permutations[idx_perms,2]
    curr_batch_size = permutations[idx_perms,3]
    curr_batch_size_np=np.array(curr_batch_size, dtype=int)
    #print(type(curr_batch_size))
    #print(type(curr_batch_size_np))
    #print(curr_batch_size_np.shape)
    #print(curr_batch_size)
    curr_optimizer = permutations[idx_perms,4]
    print('\n')
    print('current lr: ' +str(curr_lr))
    print('current dropout: ' +str(curr_drop_out))
    print('current init: ' +str(curr_init))
    print('current batch size: ' +str(curr_batch_size_np))
    print('current optimizer: ' +str(curr_optimizer))
    
    #print(f'current lr: {curr_lr}:')
    #print(f'current drop out: {curr_drop_out}')
    #print(f'current mode: {curr_init}')
    #print(f'current batch size: {curr_batch_size}')
    #print(f'current optimizer: {curr_optimizer}')
    #print('\n')
    model = unet_model.unet(learning_rate=curr_lr.astype(np.float),drop_out=curr_drop_out.astype(np.float),weight_init_mode=curr_init, optimizer = curr_optimizer)    
        
    #Model evaluation loop
    for fold_number in range(k):
        print('Training fold' + str(fold_number))
        x_training = get_items(x_train[fold_number])
        y_training = get_items(y_train[fold_number])
        x_valid = get_items(x_validation[fold_number])
        y_valid = get_items(y_validation[fold_number])
        generator = dataGenerator(curr_batch_size_np, x_training,y_training,data_gen_args,seed = SEED)
        value= int(len(x_training)/curr_batch_size_np)
        #print(value)
        history = model.fit_generator(generator,steps_per_epoch= value,epochs=5,verbose=1,validation_data = (x_valid,y_valid), callbacks=callbacks) #callbacks=callbacks)
        scores = model.evaluate(x_valid, y_valid)
        cv_losses_temp.append(scores[0])
    cv_losses.append(np.array(cv_losses_temp).mean())
    tf.reset_default_graph()
df = pd.DataFrame({'learning rate':permutations[:,0],'drop_out':permutations[:,1],'weight_init':permutations[:,2],'batch_size':permutations[:,3],'optimizer':permutations[:,4], 'val_loss':cv_losses}) 
 
df.to_csv(os.path.join("/lustre/home/d167/s1137563/Paolo_repository/unet","output_file.csv"),encoding='utf-8', index=False)
 

min_loss_conf = df.iloc[df['val_loss'].idxmin()]
print(min_loss_conf)





