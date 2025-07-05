# MatchPointAI Backend Server

A FastAPI-based backend service for tennis match prediction using machine learning. This server provides REST API endpoints for predicting tennis match outcomes based on player statistics, playing styles, and surface preferences.

## 🏗️ Architecture

The backend consists of several key components:

- **FastAPI Web Server**: RESTful API endpoints for match predictions
- **XGBoost ML Model**: Trained model for predicting match outcomes (67.95% CV accuracy, 74.50% ROC AUC)
- **PostgreSQL Database**: Player statistics and historical data storage
- **AWS Lambda Deployment**: Serverless deployment with Docker containerization

## 📁 Back-end Server Structure

```
server/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration for AWS Lambda
├── deploy-lambda.sh       # AWS Lambda deployment script
├── model/                 # Machine learning model components
│   ├── predict_matchup.py     # Prediction logic
│   ├── train_model.py         # Model training script
│   ├── create_feature_set.py  # Feature engineering
│   └── xgb_model_6795_7450.joblib  # Trained XGBoost model
├── dataset_creation/      # Data processing and feature engineering
│   ├── data_aggregation.py    # Data aggregation scripts
│   └── manage_features.py     # Feature management utilities
└── data/                  # Tennis match datasets
    ├── main/              # Historical ATP match data (1968-2024)
    ├── computed/          # Processed and aggregated data
    └── atp_players.csv    # Player information
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database
- AWS CLI (for deployment)

### Local Development Setup

1. **Clone and navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file with the following variables:
   ```env
   POSTGRES_NEON_STRING=postgresql://username:password@host:port/database
   PLAYER_TABLE=your_player_table_name
   ```

5. **Run the development server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`

## 📚 API Endpoints

### Health Check
- **GET** `/` - Ping endpoint to check server status
  ```json
  {
    "Status": "MatchPointAI pinging back!"
  }
  ```

### Players
- **GET** `/players` - Get list of all available players
  ```json
  {
    "players": ["Roger Federer", "Rafael Nadal", "Novak Djokovic", ...]
  }
  ```

### Match Prediction
- **POST** `/predict` - Predict match outcome
  ```json
  {
    "p1": "Roger Federer",
    "p2": "Rafael Nadal", 
    "surface": "Clay"
  }
  ```
  
  **Response:**
  ```json
  {
    "winner": "Rafael Nadal",
    "win_probability": 68.45
  }
  ```

### Player Statistics
- **GET** `/get_player_stats?name={player_name}` - Get detailed player statistics

## 🤖 Machine Learning Model

### Model Performance
- **Cross-Validation Accuracy**: 67.95%
- **ROC AUC Score**: 74.50%
- **Algorithm**: XGBoost Classifier

### Features Used
The model uses 17 key player statistics to make predictions:

- **Serving Statistics**: First serve percentage, first serve points won, second serve points won
- **Aces & Double Faults**: Average aces per match, average double faults per match
- **Break Points**: Break points faced and saved percentages
- **Relative Performance**: Player performance relative to tour averages
- **Surface Win Rates**: Win percentages on different court surfaces
- **Physical Attributes**: Player height and playing hand

### Prediction Process
1. Retrieve player statistics from database
2. Calculate feature differences between players
3. Apply surface-specific win rate adjustments
4. Generate prediction using trained XGBoost model
5. Return winner and confidence probability

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t matchpoint-ai .
```

### Run Locally with Docker
```bash
docker run -p 8000:8000 matchpoint-ai
```

## ☁️ AWS Lambda Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- ECR repository created
- Lambda function configured

### Deploy to AWS Lambda
```bash
chmod +x deploy-lambda.sh
./deploy-lambda.sh
```

The deployment script will:
1. Build the Docker image
2. Tag and push to Amazon ECR
3. Update the Lambda function with the new image

### Configuration
Update `deploy-lambda.sh` with your AWS configuration:
- `AWS_REGION`: Your AWS region
- `ACCOUNT_ID`: Your AWS account ID
- `REPO_NAME`: ECR repository name
- `LAMBDA_FUNCTION_NAME`: Lambda function name

## 📊 Data Sources

The system uses comprehensive tennis data including:
- **ATP Match Data**: Historical match results from 1968-2024
- **Player Statistics**: Career performance metrics
- **Surface Performance**: Court-specific win rates
- **Rolling Statistics**: Recent form and performance trends

## 🔧 Development

### Adding New Features
1. Create feature engineering scripts in `dataset_creation/`
2. Update model training in `model/train_model.py`
3. Modify prediction logic in `model/predict_matchup.py`
4. Add new API endpoints in `main.py`

### Model Retraining
```bash
cd model
python train_model.py
```

### Testing
```bash
# Test prediction functionality
python -c "from model.predict_matchup import predict_matchup; print(predict_matchup('Roger Federer', 'Rafael Nadal', 'Clay'))"
```

## 🛠️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `POSTGRES_NEON_STRING` | PostgreSQL connection string | Yes |
| `PLAYER_TABLE` | Database table name for player data | Yes |

## 📝 License

This project is part of the MatchPointAI tennis prediction system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues, please refer to the main project documentation or create an issue in the repository.
