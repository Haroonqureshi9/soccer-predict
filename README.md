# soccer-predict

A machine-learning model that predicts the outcome of international football matches. Give it two teams and it returns win/draw/loss probabilities.

## What it does

- Trains on 49,000+ international matches from 1872 to 2025
- Learns each team's strength from their historical win rate
- Predicts home win / draw / away win probabilities for any matchup
- Runs as an interactive terminal app — type two teams, get a forecast

## Example
Home team: brazil

Away team: germany
Brazil (home) vs Germany (away)

away_win: 18.7%

draw: 26.5%

home_win: 54.8%

## How it works

- **Data**: international match results (date, teams, scores, venue)
- **Features**: home and away team win rates, neutral-venue flag
- **Model**: logistic regression (scikit-learn)
- **Accuracy**: ~56%, compared to a 49% baseline of always guessing the home team

## Built with

- Python 3
- pandas (data handling)
- scikit-learn (machine learning)

## Structure

- `explore.py` — loads and inspects the raw dataset
- `features.py` — builds the result label (home_win/draw/away_win)
- `model.py` — trains the model and runs the interactive predictor

## Setup

1. Install the libraries: `pip install -r requirements.txt`
2. Download the dataset as `results.csv` (international football results)
3. Run it: `python3 model.py`

## Possible improvements

- Time-aware features to avoid data leakage
- Recent form (last 5 matches) instead of all-time win rate
- Compare against other models (Random Forest)