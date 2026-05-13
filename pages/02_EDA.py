import streamlit as st
st.set_page_config(page_title='ML Web App: EDA', page_icon= "📊", initial_sidebar_state='expanded', layout='wide')

import pandas as pd
import plotly.express as px

cleaning = st.container()
explore = st.container()

with cleaning:
    st.write("## Data Cleaning🧹")
    st.write('The original dataset was scraped from a Spotify playlist provided on Kaggle! This is\
             a snippet of the given raw dataset.')
    
    spotify_data = pd.read_csv('./data/spotify_data.csv')
    st.dataframe(spotify_data.head(3))
    st.write('Basic cleaning such as dropping irrelevant columns (`track_id`,`track_name`, and `album_name`) \
             and removing null values was done.')
    st.write("Most importantly, I extracted rows of the k-pop `track_genre` and dropped rows that included\
              collaborations between 2 or more artists because I wanted to see how artists fared on their own.\
              I continued inspecting the dataset and found out that there were many non-korean artists so I removed\
             them.")
    
    st.write('There were still so many artists that needed to encoded before feeding the data into machine learning algorithms.\
              So, I grouped artists who appeared less than 5 times on the Spotify playlist as "Others" \
             since it was highly unlikely for them to be present in the training set. Below is the function \
             used to carry out feature engineering of the artists.')
    
    shorten_categories_code = '''def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Others'
    return categorical_map
    '''
    st.code(shorten_categories_code, language='python')

    
    non_encoded_data = pd.read_csv('./data/kpop_nonencoded_data.csv')
    # print countplot
    fig = px.bar(non_encoded_data, x='artists', color='artists')
    fig.update_layout(title='Number of times artist appeared on playlist', width=900, height=500)
    st.write('These were the results.', fig)

    # display ohe dataframe
    st.write("Followed up was one-hot encoding of the artist names, where 15 unique categories were converted to\
             numerical representation. Below is a snippet of the resulting dataframe.")
    
    clean_data = pd.read_csv('./data/kpop_clean_data.csv')
    clean_data.drop(columns=['Unnamed: 0'], inplace=True)
    st.dataframe(clean_data.sample(3))

with explore:
    st.divider()
    st.write('## Data exploration 🔍')
    fig1 = px.box(non_encoded_data, x='artists', y= 'popularity', color='artists',range_y=[40,100])
    fig1.update_layout(title='Popularity of Artists', width=900, height=500)
    st.write(fig1)
    st.write('Blackpink has the widest range of popularity levels and owns the highest recorded popularity score of about 87.')
    
    correl_matrix = clean_data.corr()
    fig2 = px.imshow(correl_matrix.values,
                x=correl_matrix.columns,
                y=correl_matrix.columns,
                color_continuous_scale='RdYlBu')

    # Customize the layout
    fig2.update_layout(title='Correlation Matrix Heatmap', width=900, height=700)
    st.write(fig2)
    st.write('Since all our features are of numeric data type, I plotted the correlation matrix of the features\
              as shown.')
    st.write('We see that some artists are positively correlated with (e.g. Twice and Stray Kids) and some artists\
                are negatively correlated with popularity (e.g. BTS, Jeon Somi).')
    st.write('Popularity is also positively correlated with how upbeat or positive a song is (e.g. valence, energy,\
            danceability, loudness) and having explicit language in the song is also positively correlated with \
            popularity.')

##########################################################
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)