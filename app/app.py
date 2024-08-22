import os
import requests
from flask import Flask, request, render_template, redirect
import pickle
from werkzeug.utils import secure_filename 

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = os.path.join(os.getcwd(), 'static')

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/mnist', methods=['POST', 'GET'])
def mnist():
    return render_template('mnist.html')


@app.route('/mnist/predict',methods=['POST', 'GET'])
def mnist_predict():
    """Grabs the input values and uses them to make prediction"""
    if request.method == 'POST':
        print(os.getcwd()) 
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        res = requests.post("http://torchserve-mar:8080/predictions/mnist", files={'data': open(img_path, 'rb')})
        prediction = res.json()

    return render_template('mnist.html', prediction_text=f'Predicted Number: {prediction}')



@app.route('/chestxray', methods=['POST', 'GET'])
def chestxray():
    return render_template('chestxray.html')


@app.route('/chestxray/predict',methods=['POST', 'GET'])
def chestxray_predict():
    """Grabs the input values and uses them to make prediction"""
    if request.method == 'POST':
        print(os.getcwd()) 
        image = request.files["file"]
        if image.filename == '':
            print("Filename is invalid")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        img_path = os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename)
        image.save(img_path)
        res = requests.post("http://torchserve-mar:8080/predictions/chestxray", files={'data': open(img_path, 'rb')})
        prediction = res.json()

        if prediction[0] ==  1.0:
            prediction = "Opacity"
        elif prediction[0] == 0.0:
            prediction = "Normal"

    return render_template('chestxray.html', prediction_text=f'Predicted Label: {prediction}')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
