from tensorflow import keras

def iou(y_true, y_pred, smooth=1.):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) - intersection + smooth)
