import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv(r'C:\RAF\4 сем\ИИ и ММО\spaceship-titanic\train.csv')
print(df.describe())
print(df.info())

missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values)

df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
try:
    df = df.drop('Cabin', axis=1)
except KeyError:
    print("Field does not exist")

print(df.head(5))
print(df.info())

scaler = MinMaxScaler()

df['HomePlanet'] = df['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
df = pd.get_dummies(df, columns=['HomePlanet'], prefix='HomePlanet')

df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mode()[0])
df = pd.get_dummies(df, columns=['CryoSleep'], prefix='CryoSleep')

df['Deck'] = df['Deck'].fillna(df['Deck'].mode()[0])
df = pd.get_dummies(df, columns=['Deck'], prefix='Deck')

df['Cabin_num'] = df['Cabin_num'].fillna(df['Cabin_num'].mode()[0])
df = pd.get_dummies(df, columns=['Cabin_num'], prefix='Cabin_num')

df['Side'] = df['Side'].fillna(df['Side'].mode()[0])
df = pd.get_dummies(df, columns=['Side'], prefix='Side')

df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
df = pd.get_dummies(df, columns=['Destination'], prefix='Destination')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'] = scaler.fit_transform(df[['Age']])

df['VIP'] = df['VIP'].fillna(df['VIP'].mode()[0])
df = pd.get_dummies(df, columns=['VIP'], prefix='VIP')

df['RoomService'] = df['RoomService'].fillna(df['RoomService'].median())
df['RoomService'] = scaler.fit_transform(df[['RoomService']])

df['FoodCourt'] = df['FoodCourt'].fillna(df['FoodCourt'].median())
df['FoodCourt'] = scaler.fit_transform(df[['FoodCourt']])

df['ShoppingMall'] = df['ShoppingMall'].fillna(df['ShoppingMall'].median())
df['ShoppingMall'] = scaler.fit_transform(df[['ShoppingMall']])

df['Spa'] = df['Spa'].fillna(df['Spa'].median())
df['Spa'] = scaler.fit_transform(df[['Spa']])

df['VRDeck'] = df['VRDeck'].fillna(df['VRDeck'].median())
df['VRDeck'] = scaler.fit_transform(df[['VRDeck']])

df['Name'] = df['Name'].fillna(df['Name'].mode()[0])
df = pd.get_dummies(df, columns=['Name'], prefix='Name')

missing_values = df.isnull().sum().sort_values(ascending=False)
print(missing_values)

df.to_csv('processed_titanic10.csv', index=False)