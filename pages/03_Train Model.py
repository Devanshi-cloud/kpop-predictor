import streamlit as st
st.set_page_config(page_title='ML Web App: Train Model', page_icon= "🔭", initial_sidebar_state='expanded', layout='wide')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
    border: none;
}

div.stButton button:hover {
    background-color: red;
    color: white;
}
</style> """, unsafe_allow_html=True)



# Load and preprocess the data
@st.cache_data(persist= True)
def load():
    data = pd.read_csv('./data/kpop_clean_data.csv')
    data['popular'] = np.where(data['popularity'] >= 70, 1, 0)
    data.drop(columns=['Unnamed: 0', 'popularity'], inplace=True)
    return data
data = load()

# Train test splitting
X = data.drop('popular', axis=1)
y = data['popular']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
class_names = ["Not Popular", "Popular"]

# plot metrics
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='PuRd')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(plt.gcf())
        st.divider()

    if "ROC Curve" in metrics_list:
        probas = svc_model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic Curve (ROC)')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.divider()

    if "Precision-Recall Curve" in metrics_list:
        probas = svc_model.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, probas[:, 1])
        pr_auc = auc(recall, precision)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='blue', label='PR curve (area = %0.2f)' % pr_auc)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        st.divider()


# page layout
st.title("Experiment with SVM Classifier!")

# Left column for user input
left_column, space, right_column = st.columns([2,0.2,2])

with left_column:
    st.subheader("Parameter Selection")
    kernel = st.radio("**Kernel**", ("rbf", "linear"), key="kernel", horizontal=True)
    st.write("")
    C = st.number_input("**C** (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    st.write("")
    gamma = st.radio('**Gamma**', ('scale', 'auto'), key="gamma", horizontal=True)
    st.divider()
    metrics = st.multiselect("**What metrics to plot?**", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    st.write("")
    button = st.button("Classify")

with space:
    st.write('')

with right_column:

    if button:
        st.subheader("Model Results")
        svc_model = SVC(C=C,kernel=kernel,gamma=gamma,probability=True)
        svc_model.fit(X_train,y_train)
        y_pred = svc_model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        st.write(f"Accuracy: {accuracy:.2%}")
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test,y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

################################################
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
