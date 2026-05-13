# Kpopular-ML-App
An interactive learning web app to review the data pipeline process and understand the models used to classify a K-pop's song popularity given its features such as audio properties and artist name.

Access the webpage [here](https://phyosandarwin-kpopular-ml-app-01-kpopular-ml-app-8ywdaf.streamlit.app/)!

### Web App pages
#### Main page
Outlines my motivations for undertaking this topic, the data pipeline process, performance results of all  5 classification models used.

#### EDA
Outlines interactive Plotly visualisations I used in my [EDA Notebook](https://github.com/phyosandarwin/song-prediction-site/blob/32ff9e2a9ae6a582a1891d5d50c95dffa373ff81/notebooks/Data%20cleaning.ipynb)

#### Train Model (Playground feature)
Experiment with the parameters of the selected SVM model and explore the various performance metric visualisations (Confusion Matrix, ROC-AUC curve, Precision-Recall curve)

#### Predict Popularity (Playground feature)
Experiment with the input features and get the result on whether given song is popular or not.

### Classification Models used:
- Logistic Regression
- Random Forest
- SVM ✅
- XGBoost
- KNN

### Further Improvements:
- Collect a larger dataset of Kpop songs over time to include more data from artists labelled as 'Others'.
- Consider data cleaning techniques such as removing potential outliers in dataset (more preferred if dataset was large enough) or feature scaling to optimise performance of machine learning algorithms

### Acknowledgements
Building Interactive plots on Streamlit: Misra Turp's [Streamlit Videos Playlist](https://www.youtube.com/playlist?list=PLM8lYG2MzHmRpyrk9_j9FW0HiMwD9jSl5)