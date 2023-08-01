# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow import keras
from keras import models
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# %%
anime_df = pd.read_csv('/Users/zahidlaguna/Downloads/anime/anime_with_synopsis.csv')
ratings = pd.read_csv('/Users/zahidlaguna/Downloads/anime/rating_complete.csv')
anime_df 

# %%
anime_df = anime_df.replace('Unknown', np.nan)
anime_df = anime_df.dropna()

# %%
print(ratings.shape)
ratings

# %%
#calculate the average rating of each anime
anime_rating = np.mean(ratings['rating'].groupby(ratings['anime_id']).mean())
anime_rating = np.ceil(anime_rating*10)/10
print('The Average anime rating', anime_rating)
print('The average rating of Dragon Ball Z is 7.6')

# %%
anime_df['content'] = anime_df['Name'] + ' ' + anime_df['Genres']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(anime_df['content'])

# %%
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X.toarray())
cosine_sim = cosine_similarity(X_norm)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_norm, X_norm, test_size=0.1, random_state=42)
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer1 = Dense(256, activation='relu')(input_layer)
dropout_layer1 = Dropout(0.2)(hidden_layer1)
hidden_layer2 = Dense(128, activation='relu')(dropout_layer1)
dropout_layer2 = Dropout(0.2)(hidden_layer2)
output_layer = Dense(X_norm.shape[1], activation='relu')(dropout_layer2) 

# %%
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

# %%
def get_input_vector(idx, norms):
   return norms[idx].reshape(1, -1) 
def recommend(anime_title, model, norms, indices, df, top_n=10):
  idx = indices[df['Name'] == anime_title].tolist()[0]
  anime_embedding = model.predict(get_input_vector(idx, norms))     
  sim_scores = cosine_similarity(anime_embedding, norms).reshape(-1)
  best_scores = np.argsort(sim_scores)[-top_n:]
  return df.iloc[best_scores]

# %%
anime_title = input("Enter an anime title: ").lower()
recommendations = recommend(anime_title, model, X_norm, anime_df.index, anime_df)
if recommendations.empty:
    print("No recommendations found for {}".format(anime_title))
else: 
    print("Recommended anime similar to {}:".format(anime_title))
    print(recommendations)


