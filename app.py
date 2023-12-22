# =[Modules dan Packages]========================

from flask import Flask,request,jsonify
from werkzeug.utils import secure_filename
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO

# =[Variabel Global]=============================

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG','.jpeg','.JPEG']

interpreter = tf.lite.Interpreter(model_path="model_steven_TFLITE.tflite")
interpreter.allocate_tensors()

NUM_CLASSES = 6
cifar10_classes = ['Bali', 'Sumatera Barat','Lombok','Palembang','Riau','Sumatera Utara']
id_classes = ['6', '5','4','3','2','1']


# =[Routing]=====================================

# [Routing untuk API]
@app.route("/", methods=['GET'])
def temp():

    return "yeay"


@app.route("/scan", methods=['GET','POST'])
def apiDeteksi():
    if request.method == 'POST':
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
        hasil_prediksi = []

        # Get File Gambar yg telah diupload pengguna
        uploaded_file = request.files['file']
        file_contents = uploaded_file.read() 
        filename = secure_filename(uploaded_file.filename)

        # Periksa apakah ada file yg dipilih untuk diupload
        if filename != '':

            # Set/mendapatkan extension dan path dari file yg diupload
            file_ext = os.path.splitext(filename)[1]

            # Periksa apakah extension file yg diupload sesuai (jpg)
            if file_ext in app.config['UPLOAD_EXTENSIONS']:

                # Resize the image to 224x224 pixels (assuming the input size of your model)
                image = Image.open(BytesIO(file_contents))
                image = image.resize((224, 224))
                # Extract R, G, and B values from each pixel
                image_array = np.array(image)
                image_array = image_array / 255.0  # Normalize to the range [0, 1]
                input_data = np.expand_dims(image_array, axis=0).astype(np.float32)  # Convert to FLOAT32

                # Perform inference
                input_tensor = interpreter.get_input_details()[0]['index']
                interpreter.set_tensor(input_tensor, input_data)
                interpreter.invoke()

                # Get the output
                output_tensor = interpreter.get_output_details()[0]['index']
                output_data = interpreter.get_tensor(output_tensor)[0]

                # Generate hasil prediksi dalam format yang diinginkan
                for i in range(NUM_CLASSES):
                    prediksi_item = {
                        "id": int(id_classes[i]),
                        "region": cifar10_classes[i],
                        "percent": float(output_data[i] * 100)
                    }
                    hasil_prediksi.append(prediksi_item)
                

                # Print results to console (optional)
                return jsonify({
                    "error": False,
                    "message": "clasification success",
                    "prediksi": hasil_prediksi
                })
            else:
                # Return hasil prediksi dengan format JSON
                return jsonify({
                    "error": True,
                    "message": file_ext + " not a permitted extension"
                })
    return jsonify({
        "error": True,
        "message": "File not found"
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": True,
        "message": "File is too large"
    })

@app.errorhandler(500)
def too_large(e):
    return jsonify({
        "error": True,
        "message": "Internal Server Error"
    })

        

# [Main]========================================

if __name__ == '__main__':
    app.run(debug=True)
        



