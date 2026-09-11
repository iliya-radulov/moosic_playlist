import pandas as pd

df = pd.read_csv("../data/5000_songs.csv")
df.columns = df.columns.str.strip()
print("Columns in your CSV:")
for i, col in enumerate(df.columns):
    print(f"{i}: '{col}'")