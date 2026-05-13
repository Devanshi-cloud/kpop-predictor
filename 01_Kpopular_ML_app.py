import streamlit as st
st.set_page_config(page_title='ML Web App: Intro/Summary', page_icon='👨‍🎤',layout='wide')
from streamlit_lottie import st_lottie
import requests
from PIL import Image

# load animation
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# header components
header_left, header_right = st.columns([3,1])
with header_left:
    st.title('KPOPular or Not? 🎶🤔')
    st.write('A simple machine learning project that uses several classification models to determine whether\
         a KPOP song is going to be popular given its input features comprising of audio properties and\
         artist name.')
with header_right:
    finger_heart = load_lottieurl('https://assets3.lottiefiles.com/temp/lf20_W9oggU.json')
    st_lottie(finger_heart, height=150)
    

# layout component summary
project_intro = st.container()
models = st.container()
findings = st.container()

# project introduction
with project_intro:
    st.subheader('Project Objectives and Plan')
    left, space, right = st.columns([1.5,0.5,1.5])
    
    with left:
        st.markdown('##### Objectives 📌')
        st.image(image="https://www.hellokpop.com/wp-content/uploads/2020/02/20200224_Spotify_K-Pop_1.png",
                 caption='Surge in popularity of Kpop on Spotify, a global music streaming platform')
        st.write("Given the rise of KPop over these few years, and as an avid Kpop/music fan myself,\
                 I wanted to explore **what is the secret recipe in producing the Kpop songs that we \
                 hear today on Spotify**. Moreover, Kpop artists are known globally for their enormous \
                 and dedicated fanbase. So, I presumed the artist name has to be one of the popularity factors.")
    with space:
        st.write('')
    with right:
        st.markdown('##### Project Steps 🐾')
        st.write("1) Data cleaning and extraction")
        st.write("2) Feature processing")
        st.write("3) Model training")
        st.write("4) Evaluate models and determine the best-performing model")
        st.write("5) Deploy web app for users to experiment with the parameters of the selected model")



with models:
    st.divider()
    st.subheader('Findings of Classification Models used')

    st.markdown("##### Overall Model Performance Metrics")
    model_results_img = Image.open('./image_results/model_performance_results.png')
    resized_model_img = model_results_img.resize((800,600))
    st.image(image=resized_model_img, caption='SVM Classifier is performing consistently well across the metrics.')
    st.write('')
    st.write('')
    with st.expander('**Logistic Regression**'):
        lr_para = 'Without hyperparameter tuning, mean CV accuracy is 0.580 and test accuracy is 0.676. After tuning\
                    the model\'s parameters such as `penalty`, `solver` type and `C` value (controls regularisation), \
                    mean CV accuracy is 0.593 but test accuracy dropped to 0.657.'
        st.write(lr_para)
        lr_cm = Image.open('./image_results/lr_cm_tuning.png')
        lr_cm= lr_cm.resize((600,300))
        st.image(image=lr_cm, caption="The number of false negatives is comparable to true positives, \
                 which may easily result in misclassification. This is undesirable.")

    with st.expander('**Random Forest Classifier**'):
        rf_para = 'Without hyperparameter tuning, mean CV accuracy is 0.583 and test accuracy is 0.610. However, tuning\
                    the model\'s parameters such as `max_depth`, `max_features` and `n_estimators` (number of trees used\
                        for learning) did not see improvements in performance.\
                            Mean CV accuracy dropped to 0.570 but test accuracy remained 0.610.'
        st.write(rf_para)
        rfc_cm = Image.open('./image_results/rfc_cm_tuning.png')
        rfc_cm= rfc_cm.resize((600,300))
        st.image(image=rfc_cm, caption="Huge overfitting because in the test set, the number of false negatives\
                  is comparable to true positives, which may easily result in misclassification. This is not desirable.")
        st.divider()
        rfc_tree = Image.open('./image_results/rfc_tree_tuning.png')
        rfc_tree= rfc_tree.resize((600,300))
        st.image(image=rfc_tree, caption= "For cutoff depth of 3, first 3 random forests seem\
                 to put more importance on the song\'s liveness and energy, as well as \
                 whether the artist is Stray Kids.")
    
    with st.expander('**SVM Classifier**'):
        svc_para = 'Without hyperparameter tuning, mean CV accuracy is 0.545 and test accuracy is 0.552. After tuning\
                    the model\'s parameters such as `kernel` and `C` value (controls regularisation), \
                    mean CV accuracy rose to 0.596 and test accuracy increased greatly to 0.676.'
        st.write(svc_para)
        svc_cm = Image.open('./image_results/svc_cm_tuning.png')
        svc_cm= svc_cm.resize((600,300))
        st.image(image=svc_cm, caption="2 songs were misclassified as popular and 32 songs were misclassified \
                 as non-popular.")

    with st.expander('**XGBoost Classifier**'):
        xgb_para = 'Without hyperparameter tuning, mean CV accuracy is 0.583 and test accuracy is 0.571. After tuning\
                    the model\'s parameters such as `eta`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`,\
                    `n_estimators`, mean CV accuracy increased to 0.593 and test accuracy improved significantly to 0.657.'
        st.write(xgb_para)
        xgb_cm = Image.open('./image_results/xgbc_cm_tuning.png')
        xgb_cm= xgb_cm.resize((600,300))
        st.image(image=xgb_cm, caption="16 songs were misclassified as popular and 28 songs were misclassified \
                 as non-popular.")
    
    with st.expander('**KNN Classifier**'):
        knn_para = 'Without hyperparameter tuning, mean CV accuracy is 0.545 and test accuracy is 0.697. After tuning\
                    the model\'s parameters such as `n_neighbors`,`weights`, `metric`, \
                    mean CV accuracy increased to 0.571 and test accuracy rose to 0.714.'
        st.write(knn_para)
        knn_cm = Image.open('./image_results/knn_cm_tuning.png')
        knn_cm= knn_cm.resize((600,300))
        st.image(image=knn_cm, caption="13 songs were misclassified as popular and 17 songs were misclassified \
                 as non-popular.")
        
st.divider()
st.markdown("### Further improvements")
st.write("* Collect a larger dataset of Kpop songs, especially for artists labelled as 'Others'.")
st.write("* Consider data cleaning techniques such as removing potential outliers in dataset (usually more preferred if\
         dataset was large enough) or feature scaling to optimise performance of machine learning algorithms")