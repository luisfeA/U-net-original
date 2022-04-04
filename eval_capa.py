import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_coef, iou
from train2 import load_data
from train2 import load_data2
from train2 import load_data3
from train2 import load_data4
from train2 import load_data5
from train2 import load_data6
from train2 import load_data7
from train2 import load_data8
from train2 import load_data9
from train2 import load_data10

H = 580
W = 780

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)   ## (1, 256, 256, 3)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x > 0.5
    x = x.astype(np.int32)
    return ori_x, x

def save_result(ori_x, ori_y, y_pred, save_path):
    line = np.ones((H, 10, 3)) * 255

    ori_y = np.expand_dims(ori_y, axis=-1) ## (256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1) ## (256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255.0

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_path, y_pred)

if __name__ == "__main__":
    create_dir("results")

    """ Dataset """
    dataset_path = "experimentos/imagenes/"
    create_dir("experimentos/resultados/experimento1")
    create_dir("experimentos/resultados/experimento2")
    create_dir("experimentos/resultados/experimento3")
    create_dir("experimentos/resultados/experimento4")
    create_dir("experimentos/resultados/experimento5")
    create_dir("experimentos/resultados/experimento6")
    create_dir("experimentos/resultados/experimento7")
    create_dir("experimentos/resultados/experimento8")
    create_dir("experimentos/resultados/experimento9")
    create_dir("experimentos/resultados/experimento10")

    create_dir("experimentos/resultados/experimento1/capa4")
    create_dir("experimentos/resultados/experimento2/capa4")
    create_dir("experimentos/resultados/experimento3/capa4")
    create_dir("experimentos/resultados/experimento4/capa4")
    create_dir("experimentos/resultados/experimento5/capa4")
    create_dir("experimentos/resultados/experimento6/capa4")
    create_dir("experimentos/resultados/experimento7/capa4")
    create_dir("experimentos/resultados/experimento8/capa4")
    create_dir("experimentos/resultados/experimento9/capa4")
    create_dir("experimentos/resultados/experimento10/capa4")

    create_dir("experimentos/resultados/experimento1/capa3-4")
    create_dir("experimentos/resultados/experimento2/capa3-4")
    create_dir("experimentos/resultados/experimento3/capa3-4")
    create_dir("experimentos/resultados/experimento4/capa3-4")
    create_dir("experimentos/resultados/experimento5/capa3-4")
    create_dir("experimentos/resultados/experimento6/capa3-4")
    create_dir("experimentos/resultados/experimento7/capa3-4")
    create_dir("experimentos/resultados/experimento8/capa3-4")
    create_dir("experimentos/resultados/experimento9/capa3-4")
    create_dir("experimentos/resultados/experimento10/capa3-4")

    create_dir("experimentos/resultados/experimento1/capa2-3-4")
    create_dir("experimentos/resultados/experimento2/capa2-3-4")
    create_dir("experimentos/resultados/experimento3/capa2-3-4")
    create_dir("experimentos/resultados/experimento4/capa2-3-4")
    create_dir("experimentos/resultados/experimento5/capa2-3-4")
    create_dir("experimentos/resultados/experimento6/capa2-3-4")
    create_dir("experimentos/resultados/experimento7/capa2-3-4")
    create_dir("experimentos/resultados/experimento8/capa2-3-4")
    create_dir("experimentos/resultados/experimento9/capa2-3-4")
    create_dir("experimentos/resultados/experimento10/capa2-3-4")

    for i in range(1, 11):
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
        """ Prediction and metrics values """

        if i == 1:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
            save_path = "experimentos/resultados/experimento1/capa4/"
            cvc_path = "experimentos/resultados/experimento1/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model1.h5")
        elif i == 2:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data2(dataset_path)
            save_path = "experimentos/resultados/experimento2/capa4/"
            cvc_path = "experimentos/resultados/experimento2/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model2.h5")
        elif i == 3:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data3(dataset_path)
            save_path = "experimentos/resultados/experimento3/capa4/"
            cvc_path = "experimentos/resultados/experimento3/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model3.h5")
        elif i == 4:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data4(dataset_path)
            save_path = "experimentos/resultados/experimento4/capa4/"
            cvc_path = "experimentos/resultados/experimento4/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model4.h5")
        elif i == 5:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data5(dataset_path)
            save_path = "experimentos/resultados/experimento5/capa4/"
            cvc_path = "experimentos/resultados/experimento5/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model5.h5")
        elif i == 6:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data6(dataset_path)
            save_path = "experimentos/resultados/experimento6/capa4/"
            cvc_path = "experimentos/resultados/experimento6/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model6.h5")
        elif i == 7:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data7(dataset_path)
            save_path = "experimentos/resultados/experimento7/capa4/"
            cvc_path = "experimentos/resultados/experimento7/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model7.h5")
        elif i == 8:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data8(dataset_path)
            save_path = "experimentos/resultados/experimento8/capa4/"
            cvc_path = "experimentos/resultados/experimento8/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model8.h5")
        elif i == 9:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data9(dataset_path)
            save_path = "experimentos/resultados/experimento9/capa4/"
            cvc_path = "experimentos/resultados/experimento9/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model9.h5")
        elif i == 10:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data10(dataset_path)
            save_path = "experimentos/resultados/experimento10/capa4/"
            cvc_path = "experimentos/resultados/experimento10/capa4/score.csv"
            """ Load Model """
            with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
                model = tf.keras.models.load_model("/home/DIINF/labello/U-net-original/files-menos-4-capa/model10.h5")

        SCORE = []
        for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
            name = x.split("/")[-1]
            print(name)
            """ Reading the image and mask """
            ori_x, x = read_image(x)
            ori_y, y = read_mask(y)
            """ Prediction """
            y_pred = model.predict(x)[0] > 0.5
            #print(y_pred)
            y_pred = np.squeeze(y_pred, axis=-1)
            #print(y_pred)
            y_pred = y_pred.astype(np.int32)
            #print(y_pred)
            save_path = save_path+f"{name}"
            #save_path = f"/home/DIINF/labello/U-net-original/experimentos/resultados/{name}"
            save_result(ori_x, ori_y, y_pred, save_path)

            """ Flattening the numpy arrays. """
            y = y.flatten()
            y_pred = y_pred.flatten()

            """ Calculating metrics values """
            acc_value = accuracy_score(y, y_pred)
            f1_value = f1_score(y, y_pred, labels=[0, 1], average="binary")
            jac_value = jaccard_score(y, y_pred, labels=[0, 1], average="binary")
            recall_value = recall_score(y, y_pred, labels=[0, 1], average="binary")
            precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
            SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

        """ Metrics values """
        score = [s[1:]for s in SCORE]
        score = np.mean(score, axis=0)
        print(f"Accuracy: {score[0]:0.5f}")
        print(f"F1: {score[1]:0.5f}")
        print(f"Jaccard: {score[2]:0.5f}")
        print(f"Recall: {score[3]:0.5f}")
        print(f"Precision: {score[4]:0.5f}")

        """ Saving all the results """
        df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
        df.to_csv(cvc_path)