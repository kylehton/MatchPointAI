# MatchPointAI

## Overview

**MatchPointAI** is a personal machine learning project that predicts the outcome of professional ATP tennis matches. Built on historical ATP match data, it leverages a custom XGBoost classification model to forecast matchups between any two players in the database.

---

## Features

- **Predict match outcomes** between any two ATP players, factoring in surface and player stats
- **Interactive web interface** for easy player and surface selection
- **Comparative player statistics**, filled in with **KNN Imputation**
- **Randomized Search Cross Validation** with **Hyperparameter Tuning** for highest accuracy
- **Modern, responsive UI** built with Next.js and Tailwind CSS
- **Backend API** powered by FastAPI and Python, serving predictions and player data

---

## Technical Details

- **Model:** XGBoost classifier trained on ATP match data.
- **Features Used:**
  - **Match Specific:**  
    - Surface (Hard, Clay, Grass, Carpet)
    - Player Ace Count
    - Player Double Fault Count
    - Player Break Points Faced/Saved
    - Serve Games
    - 1st Serve In/Won, 2nd Serve Won
  - **Player Specific:**  
    - Hand (L/R)
    - Height (cm)
    - Overall Win Rate
    - Age
- **Feature Engineering:**  
  - Most match-specific features are vectorized as the difference between Player 1 and Player 2.

---

## Project Structure

```
MatchPointAI/
  client/   # Next.js frontend (UI, API routes)
  server/   # FastAPI backend (ML model, data, API)
  README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- PostgreSQL database (for player data)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MatchPointAI.git
   cd MatchPointAI
   ```

2. **Backend (FastAPI):**
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   # Set up your .env file with database credentials
   uvicorn main:app --reload
   ```

3. **Frontend (Next.js):**
   ```bash
   cd ../client
   npm install
   npm run dev
   ```

4. **Access the app:**  
   Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Usage

- Select two players and a surface.
- Click "Predict Match" to see the predicted winner and confidence.
- (Planned) View comparative player statistics and more detailed analysis.

---

