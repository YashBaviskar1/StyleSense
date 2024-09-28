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
from tensorflow.keras.models import load_model
import glob
from numpy.linalg import norm
import random
app = Flask(__name__, )
UPLOAD_FOLDER = os.path.join('static', 'uploads')

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
classification_model  = load_model(r'static\saved-model\fashion_classifier_vgg16_2.h5')
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
            features = feature_extraction(os.path.join(UPLOAD_FOLDER , image.filename),model)
            indices = recommend(features, feature_list)
#---------------------------------------------------------------------------------------------------------
            #indices = [1163,1164,1165,10000]
            for i in range(5):
                print(filenames[indices[0][i]])
            for i in range(0,4):
                img_path = os.path.join('static', filenames[indices[0][i]])
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

@app.route('/classify', methods = ['GET', 'POST'])
def classify():
    if request.method == 'POST':
        image_file = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename) 
        print(img_path)
        image_file.save(img_path)
        predicted_label = predict_img(img_path)
        return jsonify({'category': predicted_label}) 
    return render_template("classify.html")



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

image_seen = []
def predict_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = tensorflow.keras.applications.vgg16.preprocess_input(img_array)  

    predictions = classification_model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)

    class_labels = ['dress', 'hat', 'longsleeve', 'outwear', 'pants',
                    'shirt', 'shoes', 'shorts', 'skirts', 't-shirt']

    print(f'Predicted class: {class_labels[predicted_class[0]]}')
    image_seen.append(class_labels[predicted_class[0]])
    return class_labels[predicted_class[0]]

def outfit_generation():
    outfits_dir = "static/outfits"  

 
    if not image_seen:
        return "No image detected for outfit generation."

    detected_class = image_seen[0]

    outfit_combinations = {
        "dress": ["hat", "shoes"],
        "hat": ["t-shirt", "pants"],
        "longsleeve": ["pants", "shoes"],
        "outwear": ["pants", "shirt"],
        "pants": ["shirt", "shoes"],
        "shirt": ["pants", "shoes"],
        "shoes": ["pants", "shirt"],
        "shorts": ["t-shirt", "shoes"],
        "skirts": ["t-shirt", "shoes"],
        "t-shirt": ["shorts", "shoes"]
    }


    matching_items = outfit_combinations.get(detected_class, [])

 
    selected_outfit = {}
    for item in matching_items:
        item_dir = os.path.join(outfits_dir, item)
        if os.path.exists(item_dir) and os.listdir(item_dir):  
            selected_file = random.choice(os.listdir(item_dir))
            selected_outfit[item] = os.path.join(item_dir, selected_file)
        else:
            selected_outfit[item] = "No items found in this category."

    return selected_outfit