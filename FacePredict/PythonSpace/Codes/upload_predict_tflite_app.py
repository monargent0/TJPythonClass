from flask import Flask, request , jsonify #
from werkzeug.utils import secure_filename 
import os 
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

@app.route("/uploader", methods=['GET', 'POST'])
def uploader():
    f = request.files['image'] #
    os.chdir("./Images")
    f.save(secure_filename(f.filename))
    # ----------
    os.chdir("..")
    interperter = tf.lite.Interpreter(model_path="../Data/CNN/best-gray-cnn-model.tflite")
    interperter.allocate_tensors()

    input_details = interperter.get_input_details()
    output_details = interperter.get_output_details()

    os.chdir("./Images")
    path = f.filename
    load_img = np.array(Image.open(path).resize((400,300)))
    os.chdir("..")
    input_data = np.array([load_img], dtype = np.float32)
    input_data = input_data / 255

    interperter.set_tensor(input_details[0]['index'], input_data.reshape(1,400,300,1) )
    interperter.invoke()

    output_data = interperter.get_tensor(output_details[0]['index'])

    dirNames = ['Aiden','Andrew','Cathy']
    return jsonify({'result' : dirNames[np.argmax(output_data[0])]})
    

if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 5000, debug=True)
