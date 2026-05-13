import streamlit as st
st.set_page_config(page_title= 'ML Web App: Popularity Predictor', page_icon="🤔",initial_sidebar_state='expanded', layout='wide')

import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# styles for button
css = st.markdown("""
<style>
div.stButton button:first-child {
    background-color: green;
    color: white;
    transition-duration: 0.01s;
    width: 100px;
    height: 40px;
    font-size: 14px;
    line-height: 1;
    padding: 0;
    border: None;
}

div.stButton button:hover {
    background-color: red;
    color: white;
}
</style> """, unsafe_allow_html=True)

# load features
with open("./data/feature_names.pkl","rb") as f:
    feature_names = pickle.load(f)

# load model
with open("./data/best_model.pkl", "rb") as file:

    # load the model
    svm_model = pickle.load(file)

# load animation
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


############# Page Elements ###################
# page header 
left,right = st.columns(2)
with left:
    st.header('Will the song be popular ?!?')
    st.write('Provide input for the song features.')
with right:
    prediction = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_QBU0lgPJ8M.json')
    st_lottie(prediction, width=550, height=150)

# display result message @ top of the page
top_message = st.empty()
input_fields = st.container()

# input elements
with input_fields:
    col1, col2, col3 = st.columns([1, 0.3, 1])
    with col1:
        non_encoded_data = pd.read_csv('./data/kpop_nonencoded_data.csv')
        artist_options = non_encoded_data['artists'].unique().tolist()
        artist_selection = st.selectbox('**Artist**', artist_options, 1)

        duration = st.number_input('**Song Duration** (in seconds)', format="%.4f")

        explicit = st.radio('**Explicit** (Non-explicit: 0, Explicit: 1)', [0, 1],horizontal=True)

        danceability = st.number_input('**Danceability** (Enter value between 0.0 and 1.0)', format="%.4f")

        energy = st.number_input('**Energy** (Enter value between 0.0 and 1.0)', format="%.4f")

        key = st.select_slider('**Key** (-1 if no key is detected)', [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        loudness = st.number_input('**Loudness** (in dB)', format="%.4f")

        mode = st.radio('**Mode** (Major: 1, Minor: 0)', [0, 1], horizontal=True)

    with col2:
        st.write('')

    with col3:

        speechiness = st.number_input('**Speechiness** (Enter value between 0.0 and 1.0)', format="%.4f")

        acousticness = st.number_input('**Acousticness** (Enter value between 0.0 and 1.0)', format="%.4f")

        instrumentalness = st.number_input('**Instrumentalness** (Enter value between 0.0 and 1.0)', format="%.4f")

        liveness = st.number_input('**Liveness** (Enter value between 0.0 and 1.0)', format="%.4f")

        valence = st.number_input('**Valence** (Enter value between 0.0 and 1.0)', format="%.4f")

        tempo = st.number_input('**Tempo** (in BPM)', format="%.4f")

        time_signature = st.select_slider('**Time Signature**', [3, 4, 5, 6, 7])


st.write(" ")
st.write(" ")

# button functionality
if st.button('Predict!', key='predict_button',help='Click to see if the song would be popular.'):
    # Prepare input for prediction
    # Prepare input for prediction
    input_data = pd.DataFrame(columns=feature_names)

    # Add song features to input_data
    input_data['duration'] = [duration]
    input_data['explicit'] = [explicit]
    input_data['danceability'] = [danceability]
    input_data['energy'] = [energy]
    input_data['key'] = [key]
    input_data['loudness'] = [loudness]
    input_data['mode'] = [mode]
    input_data['speechiness'] = [speechiness]
    input_data['acousticness'] = [acousticness]
    input_data['instrumentalness'] = [instrumentalness]
    input_data['liveness'] = [liveness]
    input_data['valence'] = [valence]
    input_data['tempo'] = [tempo]
    input_data['time_signature'] = [time_signature]

    # Encode the artist input
    artist_cols = [col for col in feature_names if col.startswith('artist_')]
    for artist_col in artist_cols:
        artist_name = artist_col.split('_', 1)[1]
        if artist_name != artist_selection:
            input_data[artist_col] = 0
        else:
            input_data[artist_col] = 1


    # Perform prediction
    prediction = svm_model.predict(input_data)

    # message to display
    if prediction == 1:
        top_message.success('The song is going to be popular! 😆')
        st.balloons()
    else:
        top_message.error('The song is not going to be popular... 😔')
    

####################################
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 