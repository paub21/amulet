#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import keras
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

#---- ValueError: Tensor Tensor("dense_4/Softmax:0", shape=(?, 4), dtype=float32) is not an element of this graph.
import tensorflow as tf
graph = tf.get_default_graph()
#--------------------------------------------------------------------------------------------------------------------------

img_width, img_height = 224, 224
#img_width, img_height = 150, 150 #for flower image
model_path = './models/model.h5'

# ------------------------load model for tf 1.5.0 and keras 2.1.4 ------------------------------------------- #
from keras.utils.generic_utils import CustomObjectScope
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model(model_path)
#-------------------------------------------------------------------------------------------------
#model = load_model(model_path)
#model._make_predict_function()


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x /= 255.
	#---- ValueError: Tensor Tensor("dense_4/Softmax:0", shape=(?, 4), dtype=float32) is not an element of this graph.
    with graph.as_default():
       array = model.predict(x)
	#--------------------------------------------------------------------------------------------------------------------------
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        print("Label: 01-เสือหมอบ")
    elif answer == 1:
	    print("Label: 02-เสือคำราม")
    elif answer == 2:
	    print("Label: 03-แอปเปิล")
    elif answer == 3:
	    print("Label: 04-เสือเผ่น")
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            if result == 0:
                label = ' รุ่น: เสือหมอบ'
            elif result == 1:
                label = ' รุ่น: เสือคำราม'			
            elif result == 2:
                label = ' รุ่น: แอปเปิล'
            elif result == 3:
                label = ' รุ่น: เสือเผ่น'
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=label, time = str (time.time() - start_time),imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    app.run()