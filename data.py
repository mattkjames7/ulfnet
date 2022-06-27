from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from glob import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt


def adjustData(img, mask, flag_multi_class, num_class):
    print('adjust')
    if(flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if(len(mask.shape) == 4) else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1]*new_mask.shape[2], new_mask.shape[3])
                              ) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0]*new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)


def plot_segm_history(history, things_to_plot=['iou', 'loss']):

    plt.figure(figsize=(12, 6))
    for metric_or_loss in things_to_plot:
        plt.plot(history.history[metric_or_loss], linewidth=3)

    plt.suptitle('Plot of Training Loss and IoU', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.xlabel('Number of Epochs', fontsize=20)

    plt.legend(things_to_plot, loc='center right', fontsize=15)

    plt.savefig('Plot_training_loss_and_iou.png', format='png', dpi=1000)


def testGenerator(test_path, num_image, target_size=(256, 256), as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.jpg" % i), as_gray=as_gray)
        #img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,)+img.shape)
        yield img


def saveResult(save_path, pred_im_array):
    # saves images into specified directory
    for i, item in enumerate(pred_im_array):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, str(i)+"_predict.png"), img)


def threshold_binarize(x, threshold=0.5):

    y = np.copy(x)
    y[y >= threshold] = 1
    return y


def saveResultThresholded(save_path, pred_im_array, threshold):
    # saves threshold images into specified directory
    results_thresholded = threshold_binarize(pred_im_array, threshold)
    for i, item in enumerate(results_thresholded):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, str(i) +
                               "_predict_thresholded_" + str(threshold) + ".png"), img)


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
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


def mask_to_rgba(mask, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    zeros = np.zeros((img_size, img_size))
    ones = mask.reshape(img_size, img_size)
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
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
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
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(
        cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("test image", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction thresholded", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay groundtruth prediction", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 3].imshow(mask_to_red(zero_pad_mask(
                mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(
                mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    plt.savefig('predictions_overlay_repeat.png', format='png', dpi=1000)
