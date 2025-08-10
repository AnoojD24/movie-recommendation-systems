import React, { useState, useEffect, createContext, useContext } from 'react';
import { Search, Star, User, Film, Heart, LogOut, Home, TrendingUp } from 'lucide-react';

// Types
interface User {
  id: number;
  username: string;
  email: string;
}

interface Movie {
  id: number;
  title: string;
  genres: string;
  year: number;
  imdb_rating: number;
  overview: string;
  poster_url: string;
  avg_rating?: number;
  rating_count?: number;
}

interface Rating {
  rating_id: number;
  movie: Movie;
  rating: number;
  timestamp: string;
}

interface AuthContextType {
  user: User | null;
  login: (username: string, password: string) => Promise<boolean>;
  register: (username: string, email: string, password: string) => Promise<boolean>;
  logout: () => void;
}

// API Service
const API_BASE = 'http://localhost:5000/api';

const api = {
  async request(endpoint: string, options: RequestInit = {}) {
    const url = `${API_BASE}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };
    
    try {
      const response = await fetch(url, config);
      const data = await response.json();
      return { success: response.ok, data };
    } catch (error) {
      console.error('API Error:', error);
      return { success: false, data: { error: 'Network error' } };
    }
  },

  auth: {
    async login(username: string, password: string) {
      return api.request('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password }),
      });
    },
    
    async register(username: string, email: string, password: string) {
      return api.request('/auth/register', {
        method: 'POST',
        body: JSON.stringify({ username, email, password }),
      });
    },
  },

  movies: {
    async getMovies(page = 1, search = '', genre = '') {
      const params = new URLSearchParams({ 
        page: page.toString(),
        ...(search && { search }),
        ...(genre && { genre })
      });
      return api.request(`/movies?${params}`);
    },
    
    async getMovie(id: number) {
      return api.request(`/movies/${id}`);
    },
    
    async rateMovie(movieId: number, userId: number, rating: number) {
      return api.request(`/movies/${movieId}/rate`, {
        method: 'POST',
        body: JSON.stringify({ user_id: userId, rating }),
      });
    },
  },

  recommendations: {
    async getRecommendations(userId: number, type = 'hybrid') {
      return api.request(`/recommendations/${userId}?type=${type}`);
    },
  },

  users: {
    async getUserRatings(userId: number) {
      return api.request(`/users/${userId}/ratings`);
    },
  },
};

// Auth Context
const AuthContext = createContext<AuthContextType | null>(null);

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const login = async (username: string, password: string): Promise<boolean> => {
    const { success, data } = await api.auth.login(username, password);
    if (success && data.user) {
      setUser(data.user);
      localStorage.setItem('user', JSON.stringify(data.user));
      return true;
    }
    return false;
  };

  const register = async (username: string, email: string, password: string): Promise<boolean> => {
    const { success } = await api.auth.register(username, email, password);
    return success;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
  };

  return (
    <AuthContext.Provider value={{ user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
};

// Components
const StarRating: React.FC<{
  rating: number;
  onRate?: (rating: number) => void;
  interactive?: boolean;
  size?: 'sm' | 'md' | 'lg';
}> = ({ rating, onRate, interactive = false, size = 'md' }) => {
  const [hoverRating, setHoverRating] = useState(0);
  
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  return (
    <div className="flex space-x-1">
      {[1, 2, 3, 4, 5].map((star) => (
        <Star
          key={star}
          className={`${sizeClasses[size]} ${
            interactive ? 'cursor-pointer' : ''
          } ${
            star <= (hoverRating || rating)
              ? 'fill-yellow-400 text-yellow-400'
              : 'text-gray-300'
          }`}
          onClick={() => interactive && onRate?.(star)}
          onMouseEnter={() => interactive && setHoverRating(star)}
          onMouseLeave={() => interactive && setHoverRating(0)}
        />
      ))}
    </div>
  );
};

const MovieCard: React.FC<{ movie: Movie; onRate?: (rating: number) => void }> = ({ movie, onRate }) => {
  const { user } = useAuth();

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow">
      <div className="aspect-[2/3] bg-gray-200">
        {movie.poster_url ? (
          <img
            src={movie.poster_url}
            alt={movie.title}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-full">
            <Film className="w-12 h-12 text-gray-400" />
          </div>
        )}
      </div>
      
      <div className="p-4">
        <h3 className="font-semibold text-lg mb-1 line-clamp-2">{movie.title}</h3>
        <p className="text-sm text-gray-600 mb-2">{movie.year} • {movie.genres}</p>
        
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <StarRating rating={movie.imdb_rating / 2} size="sm" />
            <span className="text-sm text-gray-600">
              {movie.imdb_rating?.toFixed(1)}
            </span>
          </div>
          {movie.avg_rating && (
            <span className="text-sm text-blue-600">
              Avg: {movie.avg_rating} ({movie.rating_count} ratings)
            </span>
          )}
        </div>
        
        {movie.overview && (
          <p className="text-sm text-gray-700 mb-3 line-clamp-3">{movie.overview}</p>
        )}
        
        {user && onRate && (
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">Rate:</span>
            <StarRating
              rating={0}
              onRate={onRate}
              interactive
              size="sm"
            />
          </div>
        )}
      </div>
    </div>
  );
};

const Navbar: React.FC<{ activeTab: string; setActiveTab: (tab: string) => void }> = ({ activeTab, setActiveTab }) => {
  const { user, logout } = useAuth();

  const navItems = [
    { id: 'home', label: 'Movies', icon: Home },
    { id: 'recommendations', label: 'For You', icon: Heart },
    { id: 'trending', label: 'Trending', icon: TrendingUp },
    { id: 'profile', label: 'Profile', icon: User },
  ];

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-4">
            <Film className="w-8 h-8 text-blue-600" />
            <h1 className="text-xl font-bold text-gray-900">MovieRec</h1>
          </div>
          
          {user && (
            <div className="flex items-center space-x-6">
              {navItems.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id)}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    activeTab === id
                      ? 'bg-blue-100 text-blue-700'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </button>
              ))}
              
              <div className="flex items-center space-x-3 pl-4 border-l">
                <span className="text-sm text-gray-700">Hi, {user.username}!</span>
                <button
                  onClick={logout}
                  className="flex items-center space-x-1 text-gray-600 hover:text-red-600 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

const LoginForm: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { login, register } = useAuth();

  const handleSubmit = async (e?: React.FormEvent | React.KeyboardEvent) => {
    e?.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        const success = await login(formData.username, formData.password);
        if (!success) {
          setError('Invalid credentials');
        }
      } else {
        const success = await register(formData.username, formData.email, formData.password);
        if (success) {
          setIsLogin(true);
          setFormData({ username: '', email: '', password: '' });
          alert('Registration successful! Please log in.');
        } else {
          setError('Registration failed');
        }
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <Film className="mx-auto h-12 w-12 text-blue-600" />
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">
            {isLogin ? 'Sign in to your account' : 'Create your account'}
          </h2>
        </div>
        
        <div className="mt-8 space-y-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                Username
              </label>
              <input
                id="username"
                type="text"
                required
                value={formData.username}
                onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
                className="mt-1 appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                placeholder="Username"
              />
            </div>
            
            {!isLogin && (
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
                  className="mt-1 appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                  placeholder="Email address"
                />
              </div>
            )}
            
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
              <input
                id="password"
                type="password"
                required
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                onKeyPress={(e) => e.key === 'Enter' && handleSubmit(e)}
                className="mt-1 appearance-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                placeholder="Password"
              />
            </div>
          </div>

          {error && (
            <div className="text-red-600 text-sm text-center">{error}</div>
          )}

          <div>
            <button
              onClick={handleSubmit}
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {loading ? 'Loading...' : (isLogin ? 'Sign in' : 'Sign up')}
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-blue-600 hover:text-blue-500 text-sm"
            >
              {isLogin
                ? "Don't have an account? Sign up"
                : 'Already have an account? Sign in'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const MoviesPage: React.FC = () => {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const { user } = useAuth();

  const loadMovies = async () => {
    setLoading(true);
    const { success, data } = await api.movies.getMovies(page, search);
    if (success) {
      setMovies(data.movies);
      setTotalPages(data.pages);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadMovies();
  }, [page, search]);

  const handleRate = async (movieId: number, rating: number) => {
    if (!user) return;
    
    const { success } = await api.movies.rateMovie(movieId, user.id, rating);
    if (success) {
      alert('Rating saved!');
      // Refresh movies to show updated ratings
      loadMovies();
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Discover Movies</h1>
        
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search movies..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
            {movies.map((movie) => (
              <MovieCard
                key={movie.id}
                movie={movie}
                onRate={(rating) => handleRate(movie.id, rating)}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="flex justify-center mt-8 space-x-2">
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((pageNum) => (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`px-3 py-1 rounded ${
                    page === pageNum
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {pageNum}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
};

const RecommendationsPage: React.FC = () => {
  const [recommendations, setRecommendations] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [recType, setRecType] = useState<'hybrid' | 'collaborative' | 'content' | 'popular'>('hybrid');
  const { user } = useAuth();

  const loadRecommendations = async () => {
    if (!user) return;
    
    setLoading(true);
    const { success, data } = await api.recommendations.getRecommendations(user.id, recType);
    if (success) {
      setRecommendations(data.recommendations);
    }
    setLoading(false);
  };

  useEffect(() => {
    loadRecommendations();
  }, [recType, user]);

  const handleRate = async (movieId: number, rating: number) => {
    if (!user) return;
    
    const { success } = await api.movies.rateMovie(movieId, user.id, rating);
    if (success) {
      alert('Rating saved!');
    }
  };

  if (!user) return null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Recommendations for You</h1>
        
        <div className="flex space-x-2 mb-6">
          {[
            { value: 'hybrid', label: 'Best Match' },
            { value: 'collaborative', label: 'Users Like You' },
            { value: 'content', label: 'Similar Movies' },
            { value: 'popular', label: 'Popular' },
          ].map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setRecType(value as any)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                recType === value
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : recommendations.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
          {recommendations.map((movie) => (
            <MovieCard
              key={movie.id}
              movie={movie}
              onRate={(rating) => handleRate(movie.id, rating)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <Heart className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-xl font-medium text-gray-900 mb-2">No recommendations yet</h3>
          <p className="text-gray-600">Rate some movies to get personalized recommendations!</p>
        </div>
      )}
    </div>
  );
};

const ProfilePage: React.FC = () => {
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();

  useEffect(() => {
    const loadRatings = async () => {
      if (!user) return;
      
      setLoading(true);
      const { success, data } = await api.users.getUserRatings(user.id);
      if (success) {
        setRatings(data);
      }
      setLoading(false);
    };

    loadRatings();
  }, [user]);

  if (!user) return null;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Your Profile</h1>
        <p className="text-gray-600">Welcome back, {user.username}!</p>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Your Ratings ({ratings.length})</h2>
        </div>
        
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : ratings.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {ratings.map((rating) => (
              <div key={rating.rating_id} className="px-6 py-4 flex items-center space-x-4">
                <div className="w-16 h-24 bg-gray-200 rounded flex-shrink-0">
                  {rating.movie.poster_url ? (
                    <img
                      src={rating.movie.poster_url}
                      alt={rating.movie.title}
                      className="w-full h-full object-cover rounded"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full">
                      <Film className="w-6 h-6 text-gray-400" />
                    </div>
                  )}
                </div>
                
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-gray-900 truncate">{rating.movie.title}</h3>
                  <p className="text-sm text-gray-600">{rating.movie.year} • {rating.movie.genres}</p>
                  <div className="flex items-center space-x-2 mt-1">
                    <StarRating rating={rating.rating} size="sm" />
                    <span className="text-sm text-gray-600">
                      Rated {rating.rating}/5 on {new Date(rating.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Star className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-xl font-medium text-gray-900 mb-2">No ratings yet</h3>
            <p className="text-gray-600">Start rating movies to build your profile!</p>
          </div>
        )}
      </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('home');
  const { user } = useAuth();

  if (!user) {
    return <LoginForm />;
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <MoviesPage />;
      case 'recommendations':
        return <RecommendationsPage />;
      case 'trending':
        return <RecommendationsPage />;
      case 'profile':
        return <ProfilePage />;
      default:
        return <MoviesPage />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      {renderContent()}
    </div>
  );
};

// Export wrapped with AuthProvider
export default function MovieRecommendationApp() {
  return (
    <AuthProvider>
      <App />
    </AuthProvider>
  );
}