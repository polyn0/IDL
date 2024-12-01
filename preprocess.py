
import pandas as pd


seed=1128

df = pd.read_csv('./data/bbq_raw.csv')

df = df[['example_id', 'question_index', 'question_polarity', 'context_condition', 'category', 'context', 'question', 'ans0', 'ans1', 'ans2', 'label']]

df.to_csv('./data/bbq.csv', index=False)


# def random_sample(group):
#     return group.sample(n=10, random_state=1128)

# # 'Category' 컬럼 기준으로 그룹화 후 샘플링
# result = df.groupby('category').apply(random_sample).reset_index(drop=True)




