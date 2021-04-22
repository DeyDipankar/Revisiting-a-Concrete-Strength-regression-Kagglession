from flask import Flask,render_template,request
import pickle as pkl
import numpy as np

model = pkl.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def prediction():
        features = np.array([[x for x in request.form.values()]])
        prediction = model.predict(features)
        return render_template('main.html', data = np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(host= 'localhost' , port= 5000, debug= True)


