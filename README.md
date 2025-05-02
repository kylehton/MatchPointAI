# MatchPointAI

## An machine learning application built on past ATP match data to predict matchups between professional ATP players.

### Why This
I wanted to build something that was a part of my life, and that was tennis. Even though I had only played for 2 years in high school,
it was always fun to play in matches and hit with friends. With that, I decided to apply myself, learn basic machine learning models,
and create my own custom classification model. Unfortunately, Federer is retired so I can't see him play against the new generation, but
with this, I can still see whether he would beat them or not!

### Technical Details
The model is a classification model, using gradient boost (XGBoost) to predict whether one player or the other would win during a match.

------------------ Features Include ------------------

Match Specific:
Surface (Hard, Clay, Grass)
Player Ace Count
Player Double Fault Count
Player Break Point Faced
Player Break Point Saved
Player Serve Games
Player 1st Serve In
Player 1st Serve Won
Player Second Serve Won

Player Specific:
Player Hand (L/R)
Player Height (cm)
Player Overall Win Rate (0-1)
Player Age

Match-specific features (besides surface type) are vectorized as the difference between the data [Player 1 - Player 2]

