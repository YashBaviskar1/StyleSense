from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from tabulate import tabulate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


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

def retrieve_and_print_data():
    users = User.query.all()
    user_data = [(user.id, user.firstname, user.email) for user in users]

    recommendations = SavedRecommendation.query.all()
    print("\nSaved Image Paths:")
    for rec in recommendations:
        if rec.id == 1:
            print(rec.image_path)
    recommendation_data = [(rec.id, rec.user_id, rec.image_path) for rec in recommendations]
    print("\nUsers:")
    print(tabulate(user_data, headers=["ID", "First Name", "Email"], tablefmt="grid"))

    print("\nSaved Recommendations:")
    print(tabulate(recommendation_data, headers=["ID", "User ID", "Image Path"], tablefmt="grid"))

if __name__ == "__main__":
    with app.app_context():
        retrieve_and_print_data()
