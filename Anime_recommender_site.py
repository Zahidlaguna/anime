import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Function to load data
@st.cache_data
def load_data():
    # Load the datasets
    # Adjust the file paths as necessary
    anime_df = pd.read_csv('anime_with_synopsis.csv')
    ratings = pd.read_csv('rating_complete.csv')
    return anime_df, ratings

# Function to preprocess data
@st.cache_data
def preprocess_data(anime_df):
    anime_df = anime_df.replace('Unknown', np.nan)
    anime_df = anime_df.dropna()

    # Combine 'Name' and 'Genres' into 'content'
    anime_df['content'] = anime_df['Name'] + ' ' + anime_df['Genres']

    # Vectorize the content
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(anime_df['content'])

    # Normalize the data
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X.toarray())

    return anime_df.reset_index(drop=True), X_norm, vectorizer, scaler

# Function to train the model
@st.cache_resource
def train_model(X_norm):
    # Build the autoencoder model
    input_layer = Input(shape=(X_norm.shape[1],))
    hidden_layer1 = Dense(256, activation='relu')(input_layer)
    dropout_layer1 = Dropout(0.2)(hidden_layer1)
    hidden_layer2 = Dense(128, activation='relu')(dropout_layer1)
    dropout_layer2 = Dropout(0.2)(hidden_layer2)
    output_layer = Dense(X_norm.shape[1], activation='relu')(dropout_layer2)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_norm, X_norm, epochs=50, batch_size=32, validation_split=0.1)

    return model

# Function to get input vector
def get_input_vector(idx, norms):
    return norms[idx].reshape(1, -1)

# Function to recommend anime
def recommend(anime_title, model, norms, df, top_n=10):
    try:
        idx_list = df[df['Name'].str.lower() == anime_title.lower()].index.tolist()
        if not idx_list:
            return pd.DataFrame()
        idx = idx_list[0]
        anime_embedding = model.predict(get_input_vector(idx, norms))
        sim_scores = cosine_similarity(anime_embedding, norms).reshape(-1)
        # Exclude the input anime itself from recommendations
        sim_scores[idx] = -1
        best_scores = np.argsort(sim_scores)[-top_n:][::-1]
        return df.iloc[best_scores]
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

# Load and preprocess data
anime_df, ratings = load_data()
anime_df, X_norm, vectorizer, scaler = preprocess_data(anime_df)

# Train the model
model = train_model(X_norm)

# Streamlit application
st.title("Anime Recommendation System")

# Allow user to input the anime title
anime_title = st.text_input("Enter an anime title:")

# Allow user to select the number of recommendations
num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=20, value=10)

# Display recommendations when the button is clicked
if st.button('Recommend'):
    if anime_title:
        recommendations = recommend(anime_title, model, X_norm, anime_df, top_n=num_recommendations)
        if recommendations.empty:
            st.warning(f"No recommendations found for '{anime_title}'. Please check the spelling or try another title.")
        else:
            st.success(f"Recommended anime similar to '{anime_title}':")
            for idx, row in recommendations.iterrows():
                st.markdown(f"### {row['Name']}")
                st.write(f"**Genres:** {row['Genres']}")
                st.write(f"**Synopsis:** {row['sypnopsis']}")
                st.markdown("---")
    else:
        st.warning("Please enter an anime title to get recommendations.")
