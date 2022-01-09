#No usar rellenado de datos en las convoluciones
#No usar normalizacion por lotes despues de las convoluciones

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, \
    Input
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import TensorBoard
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import CustomObjectScope
import matplotlib.image as mpimg

#METRICAS*************************************************************
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


#CARGAR IMAGENES*************************************************************
def load_data(path, split=0.3):
    #images = os.listdir("/sperSegGs/images/")
    #mask = os.listdir("/sperSegGs/mask/")

    train_x = sorted(glob(os.path.join(path+"train/", "images", "*.jpg")))
    print(train_x)
    train_y = sorted(glob(os.path.join(path+"train/", "masks", "*.jpg")))

    valid_x = sorted(glob(os.path.join(path+"valid/", "images", "*.jpg")))
    valid_y = sorted(glob(os.path.join(path+"valid/", "masks", "*.jpg")))

    test_x = sorted(glob(os.path.join(path+"test/", "images", "*.jpg")))
    test_y = sorted(glob(os.path.join(path+"test/", "masks", "*.jpg")))
    #images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    #print(images)
    #masks = sorted(glob(os.path.join(path, "masks", "*.jpg")))
    #size = int(len(images) * split)

    #train_x, valid_x = train_test_split(images, test_size=size, random_state=42)
    #train_y, valid_y = train_test_split(masks, test_size=size, random_state=42)

    #train_x, test_x = train_test_split(train_x, test_size=size, random_state=42)
    #train_y, test_y = train_test_split(train_y, test_size=size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset
def tf_parse(x, y):
    H=580
    W=780
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y
def read_image(path):
    H = 580
    W = 780
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)  ## (1, 256, 256, 3)
    return ori_x, x

    # path = path.decode()
    # x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (W, H))
    # x = x / 255.0
    # x = x.astype(np.float32)
    # return x
def read_mask(path):
    H = 580
    W = 780
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

#MODELO UNET*****************************************************************
def conv_double(image, out_c):
    conv = Conv2D(out_c, (3, 3), activation="relu", padding="same")(image)
    #conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(out_c, (3, 3), activation="relu", padding="same")(conv)
    #conv = BatchNormalization()(conv)

    return conv


def encoder_block(image, out_c):
    x1 = conv_double(image, out_c)
    x2 = MaxPool2D((2, 2))(x1)
    return x1, x2


def decoder_block(image, skip_features, out_c):
    x = Conv2DTranspose(out_c, (2, 2), strides=(2, 2), padding="same")(image)
    y = crop_img(skip_features, x)
    x = Concatenate()([x, y])
    x = conv_double(x, out_c)
    return x

def crop_img(tensor, tensor_target):
    target_size = tensor_target.shape[1]
    tensor_size = tensor.shape[1]

    target_size2 = tensor_target.shape[2]
    tensor_size2 = tensor.shape[2]

    delta = tensor_size - target_size
    delta = delta // 2

    delta2 = tensor_size2 - target_size2
    delta2 = delta2 // 2

    dif = (tensor_size2-delta2) - (target_size2+delta)
    return tensor[:, delta:target_size+delta, delta:(tensor_size2-delta2)-dif, :]

def model_unet(input_shape):
    image = Input(input_shape)
    down_conv_1, max_pool_1 = encoder_block(image, 64)
    down_conv_2, max_pool_2 = encoder_block(max_pool_1, 128)
    down_conv_3, max_pool_3 = encoder_block(max_pool_2, 256)
    down_conv_4, max_pool_4 = encoder_block(max_pool_3, 512)
    down_conv_5 = conv_double(max_pool_4, 1024)
    print(down_conv_1.shape)
    print(down_conv_2.shape)
    print(down_conv_3.shape)
    print(down_conv_4.shape)
    print(down_conv_5.shape)

    up_conv1 = decoder_block(down_conv_5, down_conv_4, 512)
    up_conv2 = decoder_block(up_conv1, down_conv_3, 256)
    up_conv3 = decoder_block(up_conv2, down_conv_2, 128)
    up_conv4 = decoder_block(up_conv3, down_conv_1, 64)
    print(up_conv1.shape)
    print(up_conv2.shape)
    print(up_conv3.shape)
    print(up_conv4.shape)

    out = Conv2D(1, (1, 1), activation="sigmoid")(up_conv4)
    #out = resize(out, (580, 780), mode='constant', preserve_range=True)
    out = tf.image.resize(out, [580, 780])

    print(out.shape)

    model = Model(image, out, name="U-Net")
    return model


if __name__ == "__main__":
    #TODO Moficar modelo para que acepte tamaños impares y distinto largo (No cuadrados)
    model_path = "files/model.h5"
    csv_path = "files/data.csv"
    batch_size = 1
    epochs = 2
    input_shape = (580, 780, 3)
    dataset_path = "./sperSegGs/experimento1/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    metrics = [dice_coef, iou, Recall(), Precision()]
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    # model = model_unet(input_shape)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    # model.summary()
    # results = model.fit(train_dataset, validation_data=valid_dataset, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
       model = tf.keras.models.load_model("files/model.h5")
    x = mpimg.imread(test_x[0])
    ori_x, x = read_image(test_x[0])
    preds = model.predict(x)
    name = test_x[0].split("/")[-1]
    print(name)
    cv2.imwrite(f"./sperSegGs/experimento1/resultados/{name}", preds[0])
    plt.imshow(preds[0])
    plt.show()




