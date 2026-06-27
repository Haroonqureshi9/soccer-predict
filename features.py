import pandas as pd

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

print(matches[["home_team", "away_team", "home_score", "away_score", "result"]].head())
print()
print(matches["result"].value_counts())