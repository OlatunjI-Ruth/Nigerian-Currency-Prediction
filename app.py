from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import numpy as np
from keras.models import load_model
import re
import pathlib
import sys
import os
from tensorflow.keras.preprocessing import image
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
sys.path.append(os.path.abspath("./model"))
global graph, model
import tensorflow_hub as hub


app = Flask(__name__, template_folder='Template')
model = load_model('model1.h5',
                    custom_objects={'KerasLayer':hub.KerasLayer})

@app.route('/', methods=['GET'])
def index_view():
    return render_template("index.html")


@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file']
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(basepath, secure_filename(image.filename))
        image.save(image_path)
        img=image.load_img(image_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x=x/225
        x= np.expand_dims(x, axis=0)
        image_tensor = np.vstack([x])
        classes = model.predict(image_tensor)
        classes = np.argmax(classes, axis=1)
        print(classes)
        if classes==2:
          response = 'This is 50 naira note'
        elif classes==0:
          response = 'This is a 1000 naira note' 
        elif classes==7:
          response = 'This is a 500 naira note'
        elif classes==4:
          response = 'This is a 200 naira note'
        elif classes == 5:
          response = "This is a 5 naira note"
        elif classes==3:
          response = 'This is a 20 naira note'
        elif classes==6:
          response response = 'This is a 100 Naira note'
        else:
          response = 'This ia a 10 Naira note'
      return render_template("index.html", predictions=response, image=image)
    
    
  if __name__ == '__main__':
    port = int(os.getenv('PORT'))
    app.debug = True
    app.run(host='0.0.0.0', port=port)
               
