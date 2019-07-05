import os
import matplotlib.pyplot as plt
import numpy as np 
import skimage.transform as trans

from skimage.filters import threshold_otsu
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from skimage import io


def dataGenerator(batch_size,im_data ,label_data, aug_dict,image_save_prefix  = "image", 
                    save_to_dir = None,seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow(
        im_data, y=None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow(
        label_data,y=None,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
            glob_thresh = threshold_otsu(mask)
            binary_mask = mask > glob_thresh
            yield (img,binary_mask)       




def load_data_Kfold(path_X,path_Y,k):
    train_files = glob(os.path.join(path_X,'*.jpg'))
    train_labels = glob(os.path.join(path_Y,'*.jpg'))
    X_train_np = np.asarray(train_files)
    Y_train_np = np.asarray(train_labels)
    kf = KFold(n_splits=k,shuffle=True,random_state=1)
    X_valid = []
    X_train = []
    y_train = []
    y_valid = []
    for train_index, test_index in kf.split(X_train_np):
        X_train.append([np.sort(X_train_np[train_index])])
        X_valid.append([np.sort(X_train_np[test_index])])
        y_train.append([np.sort(Y_train_np[train_index])])
        y_valid.append([np.sort(Y_train_np[test_index])])
    return X_train, X_valid, y_train, y_valid    



def get_items(list_of_lists, target_dim = (256,256)):
    image_list = [] 
    flat_list = [item for list_of_lists[0] in list_of_lists for item in list_of_lists[0]]
    for j in range(len(flat_list)):
        img = io.imread(flat_list[j],as_gray = True)
        img = trans.resize(img, target_dim, mode='constant')
        image_list.append(img)
        image_np = np.asarray(image_list)
        image_np = np.expand_dims(image_np,axis=3)
    return image_np   




def test_file_reader(test_path, as_gray = True, target_dim = (256,256)):
    '''
        Reads path, resized and returns all images on specified folder
    '''
    extensions = glob(os.path.join(test_path,'*.jpg'))
    for filename in extensions:
        img = io.imread(filename,as_gray = as_gray)
        img = trans.resize(img, target_dim, mode='constant')
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img


# TO BE FIXED
def saveResult(save_path,pred_im_array): 
    #saves images into specified directory
    for i,item in enumerate(pred_im_array):
        img = item[:,:,0]
       # io.imsave(os.path.join(save_path,f"{i}_predict.png"),img)
        io.imsave(os.path.join(save_path,str(i)+"_predict.png"),img)

        
''' 
def plot_metrics(history_obj):
    fig = plt.figure()
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['acc']) 
    plt.title('model performance')  
    plt.xlabel('epoch')  
    plt.legend(['loss', 'accuracy'], loc='upper left') 
    fig.savefig('model_performance.png', dpi=1000)   
'''

def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])

def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    zeros = np.zeros((img_size,img_size))
    ones = mask.reshape(img_size,img_size)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)

def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def plot_imgs(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              nm_img_to_plot=10, 
              figsize=4,
              alpha=0.5
             ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

        
    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("overlay", fontsize=15) 
    else:
        axes[0, 2].set_title("overlay", fontsize=15) 
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()        
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.savefig('predictions_overlay.png', format='png', dpi=1000)    


#TO BE FIXED

def plot_segm_history(history, fold_number, metrics=['acc', 'val_acc', 'iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
   # plt.suptitle(f'metrics over epochs - Training Fold {fold_number}', fontsize=20)
    plt.suptitle('metrics over epochs - Training Fold'+ str(fold_number), fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    #plt.savefig(f'acc_vs_epochs_tf_{fold_number}.png', format='png', dpi=1000)
    plt.savefig('acc_vs_epochs_tf_'+str(fold_number)+'.png', format='png', dpi=1000)
    # summarize history for loss
    plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
   # plt.suptitle(f'loss over epochs - Training Fold {fold_number}', fontsize=20)
    plt.suptitle('loss over epochs - Training Fold'+ str(fold_number), fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.savefig('losses_vs_epochs_tf_'+str(fold_number)+'.png', format='png', dpi=1000)


