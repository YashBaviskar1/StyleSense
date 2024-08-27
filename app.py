from flask import Flask, render_template, url_for, request
import os
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            return render_template("recommendation.html")
    return render_template("recommendation.html")