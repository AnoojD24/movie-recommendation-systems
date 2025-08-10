# Movie Recommendation Backend

Flask backend with PyTorch-based recommendation models.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run application:
```bash
python app.py
```

The server will start on http://localhost:5000

## Features

- **Neural Collaborative Filtering**: PyTorch model for user-item recommendations
- **Content-Based Filtering**: TF-IDF vectorization of movie features
- **Hybrid Recommendations**: Combines multiple approaches
- **User Authentication**: Registration and login system
- **Rating System**: 5-star rating with real-time model updates
- **RESTful API**: Clean endpoints for frontend integration

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Movies
- `GET /api/movies` - List movies (with search/pagination)
- `GET /api/movies/{id}` - Get movie details
- `POST /api/movies/{id}/rate` - Rate a movie

### Recommendations
- `GET /api/recommendations/{user_id}` - Get personalized recommendations
  - Query params: `type=hybrid|collaborative|content|popular`

### Users
- `GET /api/users/{id}/ratings` - Get user's rating history

## Database Schema

### Users
- id (Primary Key)
- username (Unique)
- email (Unique)
- password_hash
- created_at

### Movies
- id (Primary Key)
- title
- genres
- year
- imdb_rating
- overview
- poster_url

### Ratings
- id (Primary Key)
- user_id (Foreign Key)
- movie_id (Foreign Key)
- rating (1-5)
- timestamp

## Machine Learning Models

### Neural Collaborative Filtering
```python
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dim=100):
        # User and item embeddings + deep neural network
```

### Content-Based Filtering
- Uses TF-IDF on movie genres and descriptions
- Cosine similarity for finding similar movies
- Recommends based on user's liked movies

### Hybrid Approach
- Combines collaborative and content-based scores
- Falls back to popular movies for new users
- Handles cold start problem

## Sample Data

The application initializes with 10 popular movies:
- The Shawshank Redemption (1994)
- The Godfather (1972)
- The Dark Knight (2008)
- Pulp Fiction (1994)
- Forrest Gump (1994)
- Inception (2010)
- The Matrix (1999)
- Goodfellas (1990)
- Interstellar (2014)
- The Lord of the Rings: The Fellowship of the Ring (2001)

## Model Training

The neural collaborative filtering model trains automatically:
- When the application starts
- After each new rating is submitted
- Uses 50 epochs with Adam optimizer
- MSE loss function

## Deployment

### Local Development
```bash
python app.py
```

### Production (Heroku/Railway)
```bash
# Already includes gunicorn in requirements.txt
# Create Procfile: web: gunicorn app:app
```

## Environment Variables

For production, set:
- `SECRET_KEY`: Flask secret key
- `DATABASE_URL`: Database connection string
- `TMDB_API_KEY`: For movie poster fetching (optional)

## Adding More Movies

To import MovieLens dataset:
```python
# Download MovieLens 25M dataset
# https://files.grouplens.org/datasets/movielens/ml-25m.zip

# Parse and import to database
import pandas as pd
movies_df = pd.read_csv('ml-25m/movies.csv')
# Process and insert into database
```