# app.py - Main Flask Application
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
CORS(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ratings = db.relationship('Rating', backref='user', lazy=True)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    genres = db.Column(db.String(200))
    year = db.Column(db.Integer)
    imdb_rating = db.Column(db.Float)
    overview = db.Column(db.Text)
    poster_url = db.Column(db.String(500))
    ratings = db.relationship('Rating', backref='movie', lazy=True)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id'),)

# PyTorch Neural Collaborative Filtering Model
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dim=100):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, user_id, item_id):
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        concat_vec = torch.cat([user_vec, item_vec], dim=1)
        output = self.fc_layers(concat_vec)
        return output.squeeze()

# Recommendation System Class
class RecommendationSystem:
    def __init__(self):
        self.model = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.tfidf_matrix = None
        self.content_features = None
        
    def prepare_data(self):
        """Prepare data for training"""
        users = User.query.all()
        movies = Movie.query.all()
        ratings = Rating.query.all()
        
        # Create user and movie mappings
        self.user_to_idx = {user.id: idx for idx, user in enumerate(users)}
        # Reverse mapping from index to user ID
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.movie_to_idx = {movie.id: idx for idx, movie in enumerate(movies)}
        # Reverse mapping from index to movie ID
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}
        
        # Prepare content features for content-based filtering
        movie_features = []
        for movie in movies:
            features = f"{movie.genres or ''} {movie.overview or ''}"
            movie_features.append(features)
        
        if movie_features:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            self.tfidf_matrix = tfidf.fit_transform(movie_features)
        
        return len(users), len(movies), ratings
    
    def train_neural_cf(self):
        """Train neural collaborative filtering model"""
        num_users, num_movies, ratings = self.prepare_data()
        
        if not ratings:
            return
            
        # Prepare training data
        user_ids = []
        movie_ids = []
        rating_values = []
        
        for rating in ratings:
            if rating.user_id in self.user_to_idx and rating.movie_id in self.movie_to_idx:
                user_ids.append(self.user_to_idx[rating.user_id])
                movie_ids.append(self.movie_to_idx[rating.movie_id])
                rating_values.append(rating.rating)
        
        if not user_ids:
            return
            
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids)
        movie_tensor = torch.LongTensor(movie_ids)
        rating_tensor = torch.FloatTensor(rating_values)
        
        # Initialize model
        self.model = NeuralCF(num_users, num_movies)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(50):  # Reduced epochs for demo
            optimizer.zero_grad()
            predictions = self.model(user_tensor, movie_tensor)
            loss = criterion(predictions, rating_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if not self.model or user_id not in self.user_to_idx:
            return []
            
        user_idx = self.user_to_idx[user_id]
        
        # Get all movies the user hasn't rated
        user_ratings = {r.movie_id for r in Rating.query.filter_by(user_id=user_id).all()}
        all_movies = set(self.movie_to_idx.keys())
        unrated_movies = all_movies - user_ratings
        
        if not unrated_movies:
            return []
        
        # Predict ratings for unrated movies
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for movie_id in unrated_movies:
                movie_idx = self.movie_to_idx[movie_id]
                user_tensor = torch.LongTensor([user_idx])
                movie_tensor = torch.LongTensor([movie_idx])
                pred = self.model(user_tensor, movie_tensor).item()
                predictions.append((movie_id, pred))
        
        # Sort by predicted rating and return top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in predictions[:num_recommendations]]
    
    def get_content_recommendations(self, user_id, num_recommendations=10):
        """Get recommendations using content-based filtering"""
        if self.tfidf_matrix is None:
            return []
            
        # Get user's rated movies and their average rating
        user_ratings = Rating.query.filter_by(user_id=user_id).all()
        if not user_ratings:
            return []
        
        # Find movies user liked (rating >= 4)
        liked_movies = [r.movie_id for r in user_ratings if r.rating >= 4.0]
        if not liked_movies:
            liked_movies = [r.movie_id for r in user_ratings]  # Fallback to all rated movies
        
        # Calculate content similarity
        movie_scores = {}
        for liked_movie_id in liked_movies:
            if liked_movie_id in self.movie_to_idx:
                liked_idx = self.movie_to_idx[liked_movie_id]
                similarities = cosine_similarity(
                    self.tfidf_matrix[liked_idx:liked_idx+1], 
                    self.tfidf_matrix
                ).flatten()
                
                for movie_id, movie_idx in self.movie_to_idx.items():
                    if movie_id not in [r.movie_id for r in user_ratings]:  # Not rated
                        if movie_id not in movie_scores:
                            movie_scores[movie_id] = 0
                        movie_scores[movie_id] += similarities[movie_idx]
        
        # Sort and return top recommendations
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, _ in sorted_movies[:num_recommendations]]
    
    def get_popular_recommendations(self, num_recommendations=10):
        """Get popular movies as fallback recommendations"""
        movie_ratings = db.session.query(
            Rating.movie_id,
            db.func.avg(Rating.rating).label('avg_rating'),
            db.func.count(Rating.rating).label('rating_count')
        ).group_by(Rating.movie_id).having(db.func.count(Rating.rating) >= 3).all()
        
        # Sort by average rating and rating count
        sorted_movies = sorted(
            movie_ratings, 
            key=lambda x: (x.avg_rating, x.rating_count), 
            reverse=True
        )
        
        return [movie.movie_id for movie in sorted_movies[:num_recommendations]]

# Initialize recommendation system
rec_system = RecommendationSystem()

# API Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    user = User(
        username=data['username'],
        email=data['email'],
        password_hash=generate_password_hash(data['password'])
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully', 'user_id': user.id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/movies', methods=['GET'])
def get_movies():
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '')
    genre = request.args.get('genre', '')
    
    query = Movie.query
    
    if search:
        query = query.filter(Movie.title.contains(search))
    
    if genre:
        query = query.filter(Movie.genres.contains(genre))
    
    movies = query.paginate(
        page=page, per_page=20, error_out=False
    )
    
    return jsonify({
        'movies': [{
            'id': movie.id,
            'title': movie.title,
            'genres': movie.genres,
            'year': movie.year,
            'imdb_rating': movie.imdb_rating,
            'overview': movie.overview,
            'poster_url': movie.poster_url
        } for movie in movies.items],
        'total': movies.total,
        'pages': movies.pages,
        'current_page': movies.page
    })

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    movie = Movie.query.get_or_404(movie_id)
    
    # Get average rating
    avg_rating = db.session.query(db.func.avg(Rating.rating)).filter_by(movie_id=movie_id).scalar()
    rating_count = Rating.query.filter_by(movie_id=movie_id).count()
    
    return jsonify({
        'id': movie.id,
        'title': movie.title,
        'genres': movie.genres,
        'year': movie.year,
        'imdb_rating': movie.imdb_rating,
        'overview': movie.overview,
        'poster_url': movie.poster_url,
        'avg_rating': round(avg_rating, 1) if avg_rating else None,
        'rating_count': rating_count
    })

@app.route('/api/movies/<int:movie_id>/rate', methods=['POST'])
def rate_movie(movie_id):
    data = request.get_json()
    user_id = data['user_id']
    rating_value = data['rating']
    
    # Check if user already rated this movie
    existing_rating = Rating.query.filter_by(user_id=user_id, movie_id=movie_id).first()
    
    if existing_rating:
        existing_rating.rating = rating_value
        existing_rating.timestamp = datetime.utcnow()
    else:
        rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating_value)
        db.session.add(rating)
    
    db.session.commit()
    
    # Retrain model with new data (in production, this would be done asynchronously)
    rec_system.train_neural_cf()
    
    return jsonify({'message': 'Rating saved successfully'}), 200

@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    rec_type = request.args.get('type', 'hybrid')
    
    if rec_type == 'collaborative':
        recommendations = rec_system.get_collaborative_recommendations(user_id)
    elif rec_type == 'content':
        recommendations = rec_system.get_content_recommendations(user_id)
    elif rec_type == 'popular':
        recommendations = rec_system.get_popular_recommendations()
    else:  # hybrid
        collab_recs = rec_system.get_collaborative_recommendations(user_id, 5)
        content_recs = rec_system.get_content_recommendations(user_id, 5)
        recommendations = collab_recs + content_recs
        # Remove duplicates while preserving order
        seen = set()
        recommendations = [x for x in recommendations if not (x in seen or seen.add(x))]
    
    # Get movie details
    movies = []
    for movie_id in recommendations[:10]:
        movie = Movie.query.get(movie_id)
        if movie:
            movies.append({
                'id': movie.id,
                'title': movie.title,
                'genres': movie.genres,
                'year': movie.year,
                'imdb_rating': movie.imdb_rating,
                'overview': movie.overview,
                'poster_url': movie.poster_url
            })
    
    return jsonify({
        'recommendations': movies,
        'type': rec_type
    })

@app.route('/api/users/<int:user_id>/ratings', methods=['GET'])
def get_user_ratings(user_id):
    ratings = db.session.query(Rating, Movie).join(Movie).filter(Rating.user_id == user_id).all()
    
    return jsonify([{
        'rating_id': rating.Rating.id,
        'movie': {
            'id': rating.Movie.id,
            'title': rating.Movie.title,
            'genres': rating.Movie.genres,
            'year': rating.Movie.year,
            'poster_url': rating.Movie.poster_url
        },
        'rating': rating.Rating.rating,
        'timestamp': rating.Rating.timestamp.isoformat()
    } for rating in ratings])

# Sample data initialization
def init_sample_data():
    """Initialize with sample movie data"""
    if Movie.query.count() == 0:
        sample_movies = [
            {
                'title': 'The Shawshank Redemption',
                'genres': 'Drama',
                'year': 1994,
                'imdb_rating': 9.3,
                'overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg'
            },
            {
                'title': 'The Godfather',
                'genres': 'Crime|Drama',
                'year': 1972,
                'imdb_rating': 9.2,
                'overview': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/3bhkrj58Vtu7enYsRolD1fZdja1.jpg'
            },
            {
                'title': 'The Dark Knight',
                'genres': 'Action|Crime|Drama',
                'year': 2008,
                'imdb_rating': 9.0,
                'overview': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg'
            },
            {
                'title': 'Pulp Fiction',
                'genres': 'Crime|Drama',
                'year': 1994,
                'imdb_rating': 8.9,
                'overview': 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg'
            },
            {
                'title': 'Forrest Gump',
                'genres': 'Drama|Romance',
                'year': 1994,
                'imdb_rating': 8.8,
                'overview': 'The presidencies of Kennedy and Johnson through the eyes of an Alabama man with an IQ of 75.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/arw2vcBveWOVZr6pxd9XTd1TdQa.jpg'
            },
            {
                'title': 'Inception',
                'genres': 'Action|Sci-Fi|Thriller',
                'year': 2010,
                'imdb_rating': 8.7,
                'overview': 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg'
            },
            {
                'title': 'The Matrix',
                'genres': 'Action|Sci-Fi',
                'year': 1999,
                'imdb_rating': 8.7,
                'overview': 'A computer programmer is led to fight an underground war against powerful computers who have constructed his entire reality with a system called the Matrix.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg'
            },
            {
                'title': 'Goodfellas',
                'genres': 'Biography|Crime|Drama',
                'year': 1990,
                'imdb_rating': 8.7,
                'overview': 'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/aKuFiU82s5ISJpGZp7YkIr3kCUd.jpg'
            },
            {
                'title': 'Interstellar',
                'genres': 'Adventure|Drama|Sci-Fi',
                'year': 2014,
                'imdb_rating': 8.6,
                'overview': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg'
            },
            {
                'title': 'The Lord of the Rings: The Fellowship of the Ring',
                'genres': 'Adventure|Drama|Fantasy',
                'year': 2001,
                'imdb_rating': 8.8,
                'overview': 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring.',
                'poster_url': 'https://image.tmdb.org/t/p/w500/6oom5QYQ2yQTMJIbnvbkBL9cHo6.jpg'
            }
        ]
        
        for movie_data in sample_movies:
            movie = Movie(**movie_data)
            db.session.add(movie)
        
        db.session.commit()
        print("Sample movies added to database")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        init_sample_data()
        # Train initial model
        rec_system.train_neural_cf()
    
    app.run(debug=True, port=5000)