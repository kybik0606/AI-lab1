import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv(r'C:\RAF\4 сем\ИИ и ММО\spaceship-titanic\train.csv')
print('DATA FROM DATASET:')
print(df.head(), '\n')

df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
try:
    df = df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

missing_values = df.isnull().sum().sort_values(ascending=False)
print('MISIING VALUES BEFORE FILLING:')
print(missing_values, '\n')

print('DATA FROM DATASET AFTER TRANSFORMATION:')
print(df.head(), '\n')

categorical_col = ['HomePlanet', 'CryoSleep', 'Deck', 'Cabin_num', 'Side', 'Destination', 'VIP', 'Name']
number_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

for col in categorical_col:
    df[col] = df[col].fillna(df[col].mode()[0])
    df = pd.get_dummies(df, columns=[col], prefix=col)

scaler = MinMaxScaler()
for col in number_col:
    df[col] = df[col].fillna(df[col].median())
    df[col] = scaler.fit_transform(df[[col]])

missing_values = df.isnull().sum().sort_values(ascending=False)
print('MISIING VALUES AFTER FILLING:')
print(missing_values, '\n')

df.to_csv('processed_titanic10.csv', index=False)