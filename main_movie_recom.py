import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process

# Sample movie dataset for demonstration
movies = pd.DataFrame({
    'movie_id': range(1, 51),
    'title': [
        'Inception', 'Interstellar', 'The Dark Knight', 'Memento', 'The Prestige',
        'Titanic', 'Avatar', 'The Avengers', 'Iron Man', 'Thor',
        'Captain America: The First Avenger', 'Guardians of the Galaxy', 'Black Panther', 'Doctor Strange', 'Spider-Man: Homecoming',
        'Wonder Woman', 'Aquaman', 'Justice League', 'The Flash', 'Shazam!',
        'Harry Potter and the Sorcerer\'s Stone', 'Harry Potter and the Chamber of Secrets', 
        'Harry Potter and the Prisoner of Azkaban', 'Harry Potter and the Goblet of Fire', 'Harry Potter and the Order of the Phoenix',
        'The Lord of the Rings: The Fellowship of the Ring', 'The Lord of the Rings: The Two Towers', 'The Lord of the Rings: The Return of the King', 
        'The Hobbit: An Unexpected Journey', 'The Hobbit: The Desolation of Smaug',
        'The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions', 'John Wick', 'John Wick: Chapter 2',
        'Dune', 'Blade Runner 2049', 'Star Wars: A New Hope', 'Star Wars: The Empire Strikes Back', 'Star Wars: Return of the Jedi',
        'The Lion King', 'Frozen', 'Frozen II', 'Toy Story', 'Toy Story 2',
        'Coco', 'Moana', 'Zootopia', 'Inside Out', 'Finding Nemo'
    ],
    'genres': [
        'Sci-Fi Thriller', 'Sci-Fi Drama', 'Action Thriller', 'Mystery Thriller', 'Drama Mystery',
        'Romance Drama', 'Sci-Fi Adventure', 'Action Adventure', 'Action Sci-Fi', 'Fantasy Action',
        'Action Adventure', 'Sci-Fi Comedy', 'Action Sci-Fi', 'Fantasy Adventure', 'Action Comedy',
        'Action Adventure', 'Fantasy Adventure', 'Action Sci-Fi', 'Action Comedy', 'Fantasy Adventure',
        'Fantasy Adventure', 'Fantasy Adventure', 'Fantasy Adventure', 'Fantasy Adventure', 'Fantasy Adventure',
        'Fantasy Adventure', 'Fantasy Action', 'Fantasy Adventure', 'Fantasy Adventure', 'Fantasy Adventure',
        'Sci-Fi Action', 'Sci-Fi Action', 'Sci-Fi Action', 'Action Thriller', 'Action Thriller',
        'Sci-Fi Adventure', 'Sci-Fi Thriller', 'Sci-Fi Adventure', 'Sci-Fi Adventure', 'Sci-Fi Adventure',
        'Animation Adventure', 'Animation Adventure', 'Animation Adventure', 'Animation Comedy', 'Animation Comedy',
        'Animation Adventure', 'Animation Adventure', 'Animation Adventure', 'Animation Adventure', 'Animation Comedy'
    ],
    'overview': [
        'A thief who steals corporate secrets through the use of dream-sharing technology.',
        'A team of explorers travel through a wormhole in space.',
        'When the menace known as the Joker emerges, Batman must step up.',
        'A man with short-term memory loss attempts to track down his wife\'s murderer.',
        'Two magicians engage in a competitive rivalry.',
        'A love story set against the sinking of the Titanic.',
        'A paraplegic Marine is sent to Pandora on a unique mission but becomes torn between following orders and protecting an alien world.',
        'Earth\'s mightiest heroes must come together to stop Loki.',
        'Billionaire Tony Stark becomes Iron Man after being captured in a cave.',
        'The Norse god of thunder, Thor, is cast down to Earth.',
        'Steve Rogers becomes Captain America to fight the Nazis during World War II.',
        'A group of intergalactic criminals must work together to save the universe.',
        'A Wakandan prince must fight for his throne after his father\'s death.',
        'A brilliant but arrogant surgeon learns the ways of magic.',
        'A young Peter Parker balances high school and superhero life.',
        'An Amazon princess leaves her home to fight in the war.',
        'The underwater kingdom of Atlantis is revealed.',
        'Superheroes unite to save Earth from an alien invasion.',
        'A speedster superhero battles villains across time.',
        'A young boy discovers magical powers and becomes a superhero.',
        'A young wizard discovers he is famous in the magical world.',
        'Harry returns to Hogwarts and faces new challenges.',
        'Harry learns about his family and faces new dangers.',
        'The Triwizard Tournament tests Harry\'s courage and skill.',
        'Harry must form a secret group to battle Voldemort\'s forces.',
        'A young hobbit embarks on a quest to destroy a powerful ring.',
        'The fellowship is divided as the battle against Sauron intensifies.',
        'The final battle for Middle-earth begins.',
        'Bilbo Baggins begins his unexpected journey.',
        'Bilbo faces the dragon Smaug in his quest.',
        'A computer hacker discovers the reality he lives in is a simulation.',
        'Neo must save Zion from the Machines.',
        'The final battle for the fate of humanity begins.',
        'A retired hitman seeks vengeance for his stolen car and dead dog.',
        'John Wick returns to settle an old debt.',
        'Paul Atreides must navigate the politics of a desert planet.',
        'A young officer uncovers the secrets of replicants.',
        'Luke Skywalker learns the ways of the Force.',
        'The rebels fight the Empire in a galaxy-wide war.',
        'The story of Anakin Skywalker\'s son unfolds.',
        'A lion cub learns his place in the circle of life.',
        'Anna must find Elsa, whose powers have trapped their kingdom in winter.',
        'Anna and Elsa embark on a new journey to discover the origin of Elsa\'s powers.',
        'A cowboy doll deals with the arrival of a new toy.',
        'The toys work together to save Woody from being sold.',
        'A young boy discovers the power of his family\'s history.',
        'A Polynesian girl sets sail on an epic adventure.',
        'A bunny cop teams up with a fox to solve a mystery.',
        'A young girl learns to embrace her emotions.',
        'A clownfish searches for his missing son.'
    ]
})


# Precomputed data for simplicity
movies['combined_features'] = movies['genres'] + ' ' + movies['overview']
vectorizer = CountVectorizer().fit_transform(movies['combined_features'])
similarity_matrix = cosine_similarity(vectorizer)

# Function for movie recommendations
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in movies['title'].values:
        return []
    movie_idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    recommendations = [movies.iloc[i[0]].title for i in sorted_scores]
    return recommendations

# Function to suggest movies dynamically based on user input
def suggest_movies(input_text):
    suggestions = process.extract(input_text, movies['title'].tolist(), limit=5)
    return [s[0] for s in suggestions if s[1] > 50]  # Return movies with a match score > 50

# Streamlit app setup
st.set_page_config(page_title="Movie Recommendation System", layout="wide", initial_sidebar_state="expanded")

# Sidebar with navigation
with st.sidebar:
    st.title("🎥 Movie Recommender")
    st.markdown("### Explore movies tailored to your taste")
    st.markdown("---")
    st.image("https://via.placeholder.com/200x100.png", use_container_width=True)  # Replace with your logo
    st.markdown("---")
    st.subheader("Navigation")
    nav_options = ["Home", "Recommend", "About"]
    page = st.radio("Choose a page:", nav_options)

# Home Page
if page == "Home":
    st.title("Welcome to the Movie Recommendation System")
    st.image("https://via.placeholder.com/800x400.png", use_container_width=True)  # Replace with a banner image
    st.write("""
        Discover movies tailored to your taste with our AI-powered recommendation engine.
        
        ### How It Works:
        - Enter your favorite movies in the recommendation page.
        - Receive personalized movie recommendations.
        - Enjoy exploring new titles!
    """)
    st.markdown("---")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")  # Replace with your promotional video

# Recommendation Page
elif page == "Recommend":
    st.title("Movie Recommendations 🎬")
    st.write("Type your favorite movie titles below, and we'll recommend similar ones.")

    # Input movie selection with autocomplete suggestions
    input_movie = st.text_input("Start typing a movie title:", "")
    suggestions = []
    if input_movie:
        suggestions = suggest_movies(input_movie)

    if suggestions:
        st.write("Did you mean:")
        for suggestion in suggestions:
            if st.button(suggestion):
                input_movie = suggestion

    # Number of recommendations slider
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

    # Display recommendations
    if input_movie and input_movie in movies['title'].values:
        recommendations = recommend_movies(input_movie, num_recommendations)
        if recommendations:
            st.subheader(f"Recommendations for '{input_movie}':")
            for rec in recommendations:
                st.write(f"🎥 {rec}")
        else:
            st.warning("No recommendations found for the selected movie.")
    elif input_movie and input_movie not in movies['title'].values:
        st.error("Movie not found! Please check the title or pick a suggestion.")

# About Page
elif page == "About":
    st.title("About This App")
    st.write("""
    This app provides AI-powered movie recommendations using content-based filtering.
    
    ### Features:
    - **Autocomplete Movie Search**: Type part of a movie title, and we'll suggest matches.
    - **Custom Recommendations**: Receive recommendations based on the movies you like.
    - **Modern UI**: Easy to navigate and visually appealing.
    """)
    st.image("https://via.placeholder.com/800x400.png", use_container_width=True)  # Replace with an informative image
