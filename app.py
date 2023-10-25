from __future__ import division, print_function

import base64
import sys
import os
import glob
import re
import numpy as np
from PIL import Image
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model51_vgg19.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling

    x = np.expand_dims(x, axis=0)
    x = x * 1.0 / 255

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    # preds = np.argmax(preds, axis=1)
    print("The prediction is: ", preds)
    if preds == 0:
        return render_template('index.html', preds='Corn Blight', mes='Spacing,Drip Irrigation,Proper Pruning',
                                   pest=' Mancozeb, propiconazole, azoxystrobin',
                                   link='https://www.amazon.in/AD-45-Mancozeb-75-WP-Fungicide/dp/B07JDSJY6C')
    elif preds == 1:
        return render_template('index.html', preds='Corn Gray Spot', mes='Planting Density,Spacing,Drip Irrigation',
                               pest='Mancozeb, propiconazole, azoxystrobin',
                                link="https://www.amazon.in/PROSPELL-250-AZOXYSTROBIN-MANCOZEB-Fungicide/dp/B07MFH9C1C")
    elif preds == 2:
        return render_template('index.html', preds='Corn Rust', mes='cutting shoot tips,Spacing,Drip Irrigation',
                                   pest='Propiconazole, Tebuconazole',
                                   link="https://www.bighaat.com/products/folicur")
    elif preds == 3:
        return render_template('index.html', preds='Corn Healthy',
                                   mes='No measures', pest='No pesticide',
                                   link="https://www.amazon.in/Ugaoo-Organic-Vermicompost-Fertilizer-Plants/dp/B0BDVN579S/ref=sr_1_3_sspa?hvadid=82944601526132&hvbmt=bp&hvdev=c&hvqmt=p&keywords=organic+fertilizer+for+plants&qid=1696276146&sr=8-3-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1")
    else:
        return render_template('index.html', preds='Unknown', mes='Unknown',
                               pest='Unknown',
                               link="Unknown")
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "GET":
        return "no picture"
    elif request.method == "POST":
        features = [(x) for x in request.form.values()]
        print(features)
        for i in features:
            print(i)
            print(i[23:])

        imgstring = i[23:]
        imgdata = base64.b64decode(imgstring)
        from datetime import datetime
        import os.path
        directory = './uploads/'
        filename = 'new'+ '.png'
        filepath = os.path.join(directory, filename)
        # print(f_name)

        with open(filepath, 'wb') as f:
            f.write(imgdata)
        f.close()
        print("Done")


        # Make prediction
        preds = model_predict(filepath, model)
        result = preds

        return result;
    return None


if __name__ == '__main__':
    app.run(debug=True)