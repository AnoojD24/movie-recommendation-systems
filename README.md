# movie-recommendation-systems
# Movie Recommendation System

A full-stack movie recommendation system built with Flask (PyTorch) backend and React TypeScript frontend. Features neural collaborative filtering, content-based filtering, and a modern web interface.

![Movie Recommendation System](https://via.placeholder.com/800x400/1f2937/ffffff?text=Movie+Recommendation+System)

## 🌟 Features

- **Multiple ML Models**: Neural Collaborative Filtering, Content-Based Filtering, Hybrid Approach
- **Real-time Learning**: Models retrain when users add new ratings
- **Modern UI**: Responsive React TypeScript interface with Tailwind CSS
- **Authentication**: User registration and login system
- **Interactive Rating**: 5-star rating system for movies
- **Personalized Recommendations**: Different recommendation algorithms
- **User Profiles**: Track rating history and preferences

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   Flask Backend │    │   SQLite DB     │
│   (TypeScript)   │◄──►│   (Python)      │◄──►│   (Movies,      │
│   Port 3000      │    │   Port 5000     │    │   Users, Ratings│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Setup Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 3. Setup Frontend
```bash
cd frontend
npm install
npm start
```

### 4. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## 📁 Project Structure

```
movie-recommendation-system/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── requirements.txt       # Python dependencies
│   └── README.md             # Backend documentation
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.tsx           # Main React component
│   │   ├── index.tsx         # React entry point
│   │   └── index.css         # Global styles
│   ├── package.json          # Node dependencies
│   ├── tailwind.config.js    # Tailwind configuration
│   └── README.md             # Frontend documentation
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## 🤖 Machine Learning Models

### 1. Neural Collaborative Filtering
- PyTorch implementation
- User and item embeddings
- Deep neural network for rating prediction

### 2. Content-Based Filtering
- TF-IDF vectorization of movie features
- Cosine similarity for recommendations
- Based on genres and descriptions

### 3. Hybrid Approach
- Combines collaborative and content-based methods
- Weighted scoring system
- Handles cold start problem

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | User registration |
| `/api/auth/login` | POST | User authentication |
| `/api/movies` | GET | List movies with search/filter |
| `/api/movies/{id}` | GET | Get movie details |
| `/api/movies/{id}/rate` | POST | Rate a movie |
| `/api/recommendations/{user_id}` | GET | Get recommendations |
| `/api/users/{id}/ratings` | GET | User's rating history |

## 🎯 Usage

1. **Register/Login**: Create account or sign in
2. **Rate Movies**: Click stars to rate movies (trains the model)
3. **Get Recommendations**: Visit "For You" tab for personalized suggestions
4. **Explore**: Browse movies, search, and discover new content

## 🚀 Deployment

### Backend (Railway/Heroku)
```bash
# Add Procfile: web: gunicorn app:app
pip install gunicorn
# Deploy to Railway/Heroku
```

### Frontend (Vercel/Netlify)
```bash
npm run build
# Deploy build folder to Vercel/Netlify
```

## 🛠️ Development

### Adding New Features
1. **More ML Models**: Implement deep learning models, RNNs
2. **Real Dataset**: Import MovieLens 25M dataset
3. **Advanced UI**: Movie details, trailers, social features
4. **Performance**: Redis caching, database optimization

### Testing
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

## 📊 Demo Data

The system includes 10 popular movies for testing:
- The Shawshank Redemption
- The Godfather
- The Dark Knight
- Pulp Fiction
- Forrest Gump
- And more...

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- MovieLens dataset for inspiration
- PyTorch team for excellent ML framework
- React and Flask communities

## 📞 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/movie-recommendation-system](https://github.com/yourusername/movie-recommendation-system)
