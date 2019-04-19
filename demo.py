from flask import Flask, render_template, request, flash, url_for, redirect
from werkzeug import secure_filename
import numpy as np
import pandas as pd
import cv2
import os
import pickle
import os

from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from model import *

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'RAW', 'TIF'])

application = Flask(__name__)
application.secret_key = 'dont tell anyone'
application.config.update(
    TEMPLATES_AUTO_RELOAD = True,
    MAX_CONTENT_LENGTH = 6 * 5000 * 5000,
)

def preprocess(img):

    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    return preprocess_input(img)

def get_feature(img):

    K.clear_session()
    model = cgd_model((256,256,3), 600)

    model.load_weights("./static/model_weight/40")

    img = preprocess(img)

    if len(img.shape) < 4:
        img = img[np.newaxis,:,:,:]

    intermediate_model = Model(inputs=model.input, outputs=model.layers[-3].output)

    feature = intermediate_model.predict(img)
    # feature - (1, 1024)
    return feature

def l2_normalize(v, axis=-1):

    norm = np.linalg.norm(v, axis=axis, keepdims=True)

    return np.divide(v, norm, where=norm!=0)


@application.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@application.route('/predict_part/', methods=['GET', 'POST'])
def predict_part():
    if request.method == 'POST':

        K.clear_session()

        f = request.files['file']
        img_name = secure_filename(f.filename)
        img_name = str(img_name)

        db_path = './static/db/db/'
        upload_path = './static/upload/temp/'
        temp_path = os.path.join(upload_path, img_name)

        f.save(temp_path)
        img = cv2.imread(temp_path)

        # (1,1024)
        feature = l2_normalize(get_feature(img))
        print(feature)
        with open("./static/reference/reference_part.p", "rb") as f:
            reference = pickle.load(f)

        sim_vector = np.dot(feature.reshape(-1, 1024), np.array(reference["feature"]).T)
        indice = np.argsort(sim_vector.flatten())
        indice = list(np.flip(indice)[:6])
        sorted_img = [reference["img"][i] for i in indice]
        sorted_img = [os.path.join(db_path, f) for f in sorted_img]

        return render_template('result.html', origin_img=temp_path, result=sorted_img)


@application.route('/predict_snap/', methods=['GET', 'POST'])
def predict_snap():
    if request.method == 'POST':

        K.clear_session()

        f = request.files['file']
        img_name = secure_filename(f.filename)
        img_name = str(img_name)

        db_path = './static/db/snap/'
        upload_path = './static/upload/temp/'
        temp_path = os.path.join(upload_path, img_name)

        f.save(temp_path)
        img = cv2.imread(temp_path)

        # (1,1024)
        feature = l2_normalize(get_feature(img))
        with open("./static/reference/reference_snap.p", "rb") as f:
            reference = pickle.load(f)

        sim_vector = np.dot(feature.reshape(-1, 1024), np.array(reference["feature"]).T)
        indice = np.argsort(sim_vector.flatten())
        indice = list(np.flip(indice)[:6])

        sorted_img = [reference["img"][i] for i in indice]

        sorted_img = [os.path.join(db_path, f) for f in sorted_img]

        return render_template('result.html', origin_img=temp_path, result=sorted_img)

if __name__ == '__main__':
    application.run(host='0.0.0.0')
    #application.run(debug=True)
