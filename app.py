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
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
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
    return render_template("frontpage.html")

@app.route('/frontpage')
def frontpage():
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
            return redirect(url_for('frontpage'))
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
image_seen = []
@app.route('/classify', methods = ['GET', 'POST'])
def classify():
    firstname = session.get('firstname') 
    if request.method == 'POST':
        image_file = request.files['image']
        img_path = os.path.join(UPLOAD_FOLDER, image_file.filename) 
        print(img_path)
        image_file.save(img_path)
        predicted_label = predict_img(img_path)
        image_seen.append(predicted_label)
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

@app.route('/inventory', methods=['POST', 'GET'])
def inventory():
    firstname = session.get('firstname') 
    if request.method == 'POST':
        predicted_labels = []
        uploaded_files = request.files.getlist('upload')
        if not uploaded_files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        for image_file in uploaded_files:
            img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(img_path)
            
            predicted_label = predict_img(img_path)
            predicted_labels.append(predicted_label)
        return jsonify({'categories': predicted_labels})

    return render_template("profile-inventory.html", firstname=firstname)


@app.route('/classify/generate', methods=['GET', 'POST'])
def generate_outfit():
    firstname = session.get('firstname') 
    generated_outfit = outfit_generation()  

    outfit_images = {
        item: path.replace("static\\", "").replace("\\", "/") for item, path in generated_outfit.items()
    }
    print("Generated Outfit Images:", outfit_images)
    outfit_images2 = {key: value.replace('static/', '') for key, value in outfit_images.items()}
    print(outfit_images2)
    return render_template("classify.html", firstname=firstname, outfit_images=outfit_images2)


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
    print(detected_class)
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
        "t-shirt": ["pants", "shoes"]
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

data_cleaned = pd.read_csv('cleaned_nike_data.csv')

top_10_by_avg_rating = data_cleaned.groupby('name')['avg_rating'].mean().sort_values(ascending=False).head(10)

top_10_by_num_reviews = data_cleaned['name'].value_counts().head(10)


def plot_top_10_avg_rating(top_10_data):
    fig = px.bar(top_10_data, x=top_10_data.index, y=top_10_data.values,
                 title="Top 10 Products by Average Rating", labels={'x': 'Product Name', 'y': 'Average Rating'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

def plot_top_10_num_reviews(top_10_data):
    fig = px.bar(top_10_data, x=top_10_data.index, y=top_10_data.values,
                 title="Top 10 Products by Number of Reviews", labels={'x': 'Product Name', 'y': 'Number of Reviews'},
                 template='plotly_white')
    return fig.to_html(full_html=False)


def plot_availability(data):
    availability_counts = data['availability'].value_counts()
    fig = px.bar(availability_counts, x=availability_counts.index, y=availability_counts.values,
                 title="Product Availability (In Stock vs Out Of Stock)", labels={'x': 'Availability', 'y': 'Count'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

# Plot Price Distribution using Seaborn
def plot_price_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'], bins=20, kde=True, color='blue')
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Plot Available Sizes using a bar graph
def plot_available_sizes(data):
    available_sizes_counts = data['available_sizes'].value_counts().head(10)
    fig = px.bar(available_sizes_counts, x=available_sizes_counts.index, y=available_sizes_counts.values,
                 title="Top 10 Available Sizes", labels={'x': 'Size', 'y': 'Count'},
                 template='plotly_white')
    return fig.to_html(full_html=False)

def plot_available_colors(data):
    color_counts = data['color'].value_counts().head(10)
    fig = px.pie(names=color_counts.index, values=color_counts.values, title="Top 10 Available Colors", template='plotly_white')
    return fig.to_html(full_html=False)


def create_summary():
    total_sales = data_cleaned['price'].count()
    total_revenue = data_cleaned['price'].sum()
    average_price = data_cleaned['price'].mean()
    top_product = data_cleaned.groupby('name')['price'].sum().idxmax()
    
    summary_points = [
        f"Total Sales: {total_sales}",
        f"Total Revenue: ${total_revenue:,.2f}",
        f"Average Price: ${average_price:,.2f}",
        f"Top Selling Product: {top_product}",
        f"Total Unique Products: {data_cleaned['name'].nunique()}",
        f"Total Available Stock: {data_cleaned['availability'].value_counts().get('In Stock', 0)}"
    ]
    return summary_points

@app.route('/indexb')
def indexb():
    return render_template('index-b.html')

@app.route('/transaction_analysis')
def transaction_analysis():
    avg_rating_graph = plot_top_10_avg_rating(top_10_by_avg_rating)
    num_reviews_graph = plot_top_10_num_reviews(top_10_by_num_reviews)
    return render_template('transaction_analysis.html', avg_rating_graph=avg_rating_graph, num_reviews_graph=num_reviews_graph)

@app.route('/inventory_analysis')
def inventory_analysis():
    availability_graph = plot_availability(data_cleaned)
    price_distribution_graph = plot_price_distribution(data_cleaned)
    available_sizes_graph = plot_available_sizes(data_cleaned)
    available_colors_graph = plot_available_colors(data_cleaned)
    
    return render_template('inventory_analysis.html',
                           availability_graph=availability_graph,
                           price_distribution_graph=price_distribution_graph,
                           available_sizes_graph=available_sizes_graph,
                           available_colors_graph=available_colors_graph)

@app.route('/summary')
def summary():
    summary_points = create_summary()
    return render_template('summary.html', summary_points=summary_points)

if __name__ == "__main__":
    app.run(debug=True)