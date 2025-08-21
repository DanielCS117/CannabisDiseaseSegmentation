from flask import Flask, render_template, request, redirect, url_for
import requests
import os

app = Flask(__name__)

api_url = 'http://127.0.0.1:8000/predict'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            files = {'file': (file.filename, file.stream, file.mimetype)}
            response = requests.post(api_url, files=files)

            if response.status_code == 200:
                result = response.json()
                return render_template('index.html', result=result)
            else:
                return f'Error: {response.text}'
            
    return render_template('index.html', result=None)
            
if __name__ == '__main__':
    app.run(port=5000, debug=True)
