import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

matches = pd.read_csv("results.csv")
matches = matches.dropna(subset=["home_score", "away_score"])

def get_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    else:
        return "draw"

matches["result"] = matches.apply(get_result, axis=1)

home_games = matches.groupby("home_team").size()
home_wins = matches[matches["result"] == "home_win"].groupby("home_team").size()
away_games = matches.groupby("away_team").size()
away_wins = matches[matches["result"] == "away_win"].groupby("away_team").size()

total_games = home_games.add(away_games, fill_value=0)
total_wins = home_wins.add(away_wins, fill_value=0)
win_rate = (total_wins / total_games).fillna(0)

matches["home_win_rate"] = matches["home_team"].map(win_rate)
matches["away_win_rate"] = matches["away_team"].map(win_rate)
matches["neutral_num"] = matches["neutral"].astype(int)

features = ["home_win_rate", "away_win_rate", "neutral_num"]
X = matches[features]
y = matches["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
print(f"Baseline (always home_win): 49%")

def predict_match(home_team, away_team):
    home_rate = win_rate.get(home_team, 0)
    away_rate = win_rate.get(away_team, 0)

    match_features = pd.DataFrame(
        [[home_rate, away_rate, 0]],
        columns=["home_win_rate", "away_win_rate", "neutral_num"],
    )

    probabilities = model.predict_proba(match_features)[0]
    outcomes = model.classes_

    print(f"\n{home_team} (home) vs {away_team} (away)")
    for outcome, prob in zip(outcomes, probabilities):
        print(f"  {outcome}: {prob:.1%}")


predict_match("Portugal", "Argentina")
predict_match("Brazil", "San Marino")