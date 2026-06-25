import pandas as pd

matches = pd.read_csv("results.csv")

print(f"Total matches: {len(matches)}")
print(f"Columns: {list(matches.columns)}")
print()
print(matches.head())
