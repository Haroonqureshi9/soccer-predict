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
matches["neutral_num"] = matches["neutral"].astype(int)

features = ["neutral_num"]
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