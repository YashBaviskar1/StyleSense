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
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, )
app.secret_key = 'azuitupop'  #session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
UPLOAD_FOLDER = os.path.join('static', 'uploads')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
class SavedRecommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    user = db.relationship('User', backref=db.backref('recommendations', lazy=True))
with app.app_context():
    db.create_all()


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
    firstname = session.get('firstname')
    return render_template("index.html", firstname=firstname)

@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['firstname'] = user.firstname  
            session['user_id'] = user.id  
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your email and/or password.', 'danger')
    return render_template("login.html")

@app.route('/register', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        firstname = request.form['firstname']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        new_user = User(firstname=firstname, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route('/profile')
def profile():
    firstname = session.get('firstname')
    user = User.query.filter_by(firstname=firstname).first()  
    saved_images = SavedRecommendation.query.filter_by(user_id=user.id).all() if user else []
    return render_template("profile.html", firstname= firstname,  saved_images=saved_images)

@app.route('/recommendation', methods = ['GET', 'POST'])
def recommendation():
    firstname = session.get('firstname') 
    print(firstname)
    recommended_images = []
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
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


    return render_template("recommendation.html",  firstname = firstname)
@app.route('/save_recommendation', methods=['POST'])
def save_recommendation():
    firstname = session['firstname']
    user = User.query.filter_by(firstname=firstname).first()

    if not user:
        return jsonify({"error": "User not found"}), 404

    image_path = request.json.get('image_path')

    new_recommendation = SavedRecommendation(user_id=user.id, image_path=image_path)
    db.session.add(new_recommendation)
    db.session.commit()

    return jsonify({"success": True, "message": "Image saved successfully"})
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
    firstname = session.get('firstname') 
    if request.method == 'POST':
        image_file = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename) 
        print(img_path)
        image_file.save(img_path)
        predicted_label = predict_img(img_path)
        return jsonify({'category': predicted_label}) 
    return render_template("classify.html", firstname=firstname)

@app.route('/saved_images', methods = ['GET', 'POST'])
def saved_images():
    saved_imgs = []
    firstname = session.get('firstname') 
    print(firstname)
    user_id = session.get('user_id')
    print(user_id)
    recommendations = SavedRecommendation.query.filter_by(user_id=user_id).all()
    print("\nSaved Image Paths for User ID:", user_id)
    for rec in recommendations:
        print(rec.image_path)
        img = rec.image_path.replace("static/","")
        saved_imgs.append(img)
    print(saved_imgs)
    return render_template("profile-saved-image.html", firstname=firstname, saved_imgs=saved_imgs)

@app.route('/logout')
def logout():
    session.pop('firstname', None)  
    return redirect(url_for('login'))

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


if __name__ == "__main__":
    app.run(debug=True)