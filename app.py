from flask import Flask, render_template, url_for, request, jsonify, send_from_directory
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

app = Flask(__name__, )
UPLOAD_FOLDER = os.path.join('static', 'uploads')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():
    recommended_images = []
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
#-----------> Fix the Issue : filenames.pkl fix then unedit these lines ------------------<
            # features = feature_extraction(os.path.join(UPLOAD_FOLDER , image.filename),model)
            # indices = recommend(features, feature_list)
#---------------------------------------------------------------------------------------------------------
#>---------------Static INDICES, Someone messed up the filenames.pkl real time or me like a fucking idiot got the wrong directory, i will check this later ------------------<
#--------------Yes i know critical failure from my part -----------<
            indices = [1163,1164,1165,10000]

            for i in range(0,4):
                img_path = os.path.join('static', 'datasets', 'images', f"{indices[i]}.jpg")
                recommended_images.append(img_path)
            print(recommended_images)
            return jsonify(recommended_images=recommended_images)


    return render_template("recommendation.html")
#--------------> very important to RETURN the files from the dir ------------<
@app.route('/static/datasets/images/<filename>')
def serve_image(filename):
    print("routing here seeee meeee : ", filename)
    return send_from_directory('static/datasets/images', filename)
# 'datasets/images/' + str(indices[0][i]) + '.jpg'
#static\datasets\images\1163.jpg
#"static\datasets\images\1164.jpg"
#"static\datasets\images\1165.jpg"
#static\datasets\images\10000.jpg"


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices
