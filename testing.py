import pandas as pd


def fun(x):
    x = x + 1
    x = x + 1
    # x = x * 1
    return x


df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [1, 2, 3, 4, 5]
})

results = df['A'].apply(lambda x: fun(x))

print(results)


df = pd.read_csv('temp_data/uploaded_data.csv')
df_2 = pd.read_csv('sample_data/Iris.csv')
print(df.head(10))
print(df_2.head(10))

