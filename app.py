from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
Gnb = pickle.load(open('./models/gnb.pkl', 'rb'))
@app.route("/")
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        x1 = float(request.form.get('x1'))
        x2 = float(request.form.get('x2'))
        x3 = float(request.form.get('x3'))
        x4 = float(request.form.get('x4'))

        result = Gnb.predict([[x1, x2, x3, x4]])

        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
