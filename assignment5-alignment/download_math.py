import os

import pandas as pd

TRAIN_DATA = "data/MATH-benchmark/data/train-00000-of-00001.parquet"
TEST_DATA = "data/MATH-benchmark/data/test-00000-of-00001.parquet"
COLUMNS = ["problem", "solution", "answer", "subject", "level"]

out_dir = "data/MATH-benchmark/processed/"
os.makedirs(out_dir, exist_ok=True)

train_df = pd.read_parquet(TRAIN_DATA, columns=COLUMNS)
test_df = pd.read_parquet(TEST_DATA, columns=COLUMNS)

# save as json
train_df.to_json(out_dir + "train.json", orient="records", lines=True)
test_df.to_json(out_dir + "test.json", orient="records", lines=True)

print("Data saved to", out_dir)

# =========
