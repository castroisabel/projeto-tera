import pandas as pd

df = pd.read_csv('data/fraudTest.csv')
df = df.query('trans_date_trans_time >= "2020-10-01"')
df.to_csv('data/reducedTest.csv')