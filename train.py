import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_coef, iou

H = 580
W = 780

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

def load_data(path, split=0.3):
    train_x = []
    for imagen in os.listdir("/home/DIINF/labello/U-net-original/"+path+"images-aumentadas/"):
        train_x = train_x + sorted(glob(os.path.join(path+"images-aumentadas/", imagen, "*.jpg")))


    #train_x = sorted(glob(os.path.join(path+"imagenes-aumentadas/", "Placa1-imagen1", "*.jpg")))

    train_y = []
    for mask in os.listdir("/home/DIINF/labello/U-net-original/"+path + "mask-aumentadas/"):
        train_y = train_y + sorted(glob(os.path.join(path + "mask-aumentadas/", mask, "*.jpg")))

    #train_y = sorted(glob(os.path.join(path+"mask-aumentadas/", "masks", "*.jpg")))

    valid_x = sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen12.jpg")))
    valid_x = valid_x + sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen19.jpg")))
    valid_x = valid_x + sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen20.jpg")))


    valid_y = sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen12.jpg")))
    valid_y = valid_y + sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen19.jpg")))
    valid_y = valid_y + sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen20.jpg")))


    test_x = sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen2.jpg")))
    test_x = test_x + sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen6.jpg")))
    test_x = test_x + sorted(glob(os.path.join(path + "images-original/", "Placa1-imagen10.jpg")))


    test_y = sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen2.jpg")))
    test_y = test_y + sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen6.jpg")))
    test_y = test_y + sorted(glob(os.path.join(path + "mask-original/", "Placa1-imagen10.jpg")))


    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.set_random_seed(42)

    """ Directory to save files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-4   ## 0.0001
    num_epochs = 150
    model_path = "files/model.h5"
    csv_path = "files/data.csv"

    """ Dataset """
    dataset_path = "experimentos/imagenes/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    train_x, train_y = shuffle(train_x, train_y)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # ds = (1, 2, 3, 4, 5)
    # bs = 2
    # n = len(ds)//bs = 2
    # [1, 2], [3, 4], [1]

    train_steps = (len(train_x)//batch_size)
    valid_steps = (len(valid_x)//batch_size)

    if len(train_x) % batch_size != 0:
        train_steps += 1

    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    """ Model """
    model = build_unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr), metrics=metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='loss', patience=20, restore_best_weights=False)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )