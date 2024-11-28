
import pandas as pd


df = pd.read_csv('bbq_raw.csv')

df = df[['example_id', 'question_index', 'question_polarity', 'context_condition', 'context', 'question', 'ans0', 'ans1', 'ans2', 'label']]

df.to_csv('bbq.csv', index=False)


