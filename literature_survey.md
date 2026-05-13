# Literature Survey: Predicting K-Pop Song Popularity on Spotify  

## 1. How “Popularity” Is Defined  

- **Spotify popularity index** - a numeric score (0-100) that aggregates recent plays, total plays, saves, playlist adds, and recency of activity [1].  
- **Chart-based measures** - many studies use positions or metrics derived from the Spotify Top 200 daily chart (e.g., length of chart stay, peak rank, cumulative streams) as a proxy for hit status [2].  
- **Composite popularity metrics** - some works construct several derived variables (max daily streams, sum of streams, mean streams, debut rank) to capture different aspects of success [3].

These definitions influence the labeling of “popular” vs. “non-popular” for supervised learning.

---

## 2. Feature Extraction Pipelines  

| Step | Typical Approach | Supporting Evidence |
|------|------------------|----------------------|
| **Audio descriptors** | Retrieve Spotify’s 12-dimensional audio feature set (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration, key, mode) via the Web API [1]. | |
| **Feature relevance** | Empirical studies repeatedly highlight **energy, danceability, loudness, and valence** as the strongest predictors of popularity [4]. | |
| **Additional acoustic cues** | Tempo, acousticness, and spectral characteristics are also examined across hit-song-science literature [3]. | |
| **Dataset scale** | Large-scale Spotify dumps (≈100 k tracks) provide the required metadata and audio features for model training [5]. | |
| **Artist information** | Artist name (categorical) can be encoded (e.g., one-hot or target encoding) and has been shown to carry predictive signal, especially for dominant groups such as Stray Kids [6]. | |

Typical preprocessing includes **scaling** (standardization or min-max) for distance-based models, **outlier removal** for extreme audio values, and optionally **dimensionality reduction** (PCA) when the feature space is high-dimensional.

---

## 3. Handling Limited or Imbalanced K-Pop Data  

- **Resampling techniques** - Undersampling (e.g., Tomek Links) and oversampling (SMOTE) are standard ways to balance minority “popular” classes [7].  
- **SMOTE with tree ensembles** - Combining SMOTE with Random Forest or XGBoost yields robust performance on skewed music datasets [8].  
- **Cross-validation** - Stratified k-fold CV is recommended to preserve class ratios while estimating generalization error (common across hit-song-science studies).

These strategies mitigate the risk of models learning trivial “non-popular” defaults.

---

## 4. Classification Algorithms & Reported Performance  

| Model | Typical Accuracy / CV (reported) | Observations |
|-------|----------------------------------|--------------|
| **Support Vector Machine** | Mean CV ≈ 0.545 → 0.596 after kernel/C tuning; test accuracy ↑ 0.552 → 0.676 (large gain) [3]. | Sensitive to hyper-parameters; benefits from regularisation. |
| **Logistic Regression** | CV ≈ 0.580 → 0.593 after penalty/solver tuning; test accuracy modestly declines (≈ 0.676 → 0.657) [3]. | Simple, interpretable but limited non-linear capacity. |
| **Random Forest** | CV ≈ 0.583 → 0.570 after depth/feature tuning; test accuracy stable at ≈ 0.610 [3]. | Often the top performer in music popularity studies; robust to noise. |
| **XGBoost** | CV ≈ 0.583 → 0.593 after eta/depth/weight tuning; test accuracy improves to ≈ 0.657 [3]. | Gradient boosting yields strong predictive power, especially with balanced data. |
| **K-Nearest Neighbours** | CV ≈ 0.545 → 0.571 after k/weight/metric tuning; test accuracy rises to ≈ 0.714 [3]. | Performs well on scaled feature spaces; prone to overfitting on noisy data. |
| **Ensemble comparisons** | Random Forest and XGBoost consistently rank highest across multiple studies; Random Forest often preferred for interpretability [9]. | |

Evaluation metrics beyond accuracy-**precision, recall, F1-score, ROC-AUC**-are routinely reported to capture the impact of class imbalance [10].

---

## 5. Effects of Hyper-Parameter Tuning  

- **SVM** - Adjusting kernel type and regularisation parameter C yields noticeable CV and test-set gains (≈ +0.05 CV, +0.12 test) [3].  
- **Random Forest** - Varying max-depth, max-features, and n-estimators sometimes **decreases** CV without improving test accuracy, indicating limited benefit for this dataset [3].

Thus, tuning impact is model-dependent; exhaustive grid searches may be unnecessary for robust tree ensembles.

---

## 6. Model Interpretability  

- **SHAP values** provide a unified, theoretically sound way to attribute importance to each feature in any black-box model, including Random Forest and XGBoost [11].  
- Feature-importance plots derived from SHAP often highlight **energy, liveness, and artist identity** as key drivers for K-Pop popularity, aligning with domain observations [6].

Interpretability aids stakeholder trust and informs feature-engineering decisions.

---

## 7. Extending the Predictor: Additional Factors  

- **Visual content** - Combining audio descriptors with video-frame features (e.g., colorfulness, motion intensity) improves hit prediction accuracy by ~9 % [9].  
- **Lyrics & sentiment** - Embedding-based representations of song lyrics, together with audio, raise classification AUC to > 0.90 [12].  
- **Release timing & fan-base size** - Prior work suggests debut year, group vs. solo status, and social-media follower counts correlate with streaming spikes, though quantitative models are scarce (open-ended research direction).

These modalities can be fused in multimodal deep-learning pipelines for future enhancements.

---

## 8. Best-Practice Guidelines for Model Selection & Streamlit Deployment  

1. **Data Preparation**  
   - Apply **stratified k-fold CV** (k = 5-10) to estimate performance on imbalanced classes.  
   - Use **SMOTE** or similar oversampling only on the training folds; keep the validation/test sets untouched to avoid optimistic bias [7].

2. **Model Evaluation**  
   - Report **accuracy, precision, recall, F1, and ROC-AUC**; prioritize recall/F1 when the “popular” class is rare.  
   - Conduct **ablation studies** to quantify the contribution of artist name vs. audio features.

3. **Algorithm Choice**  
   - Start with **Random Forest** for its balance of performance, robustness, and interpretability.  
   - If higher predictive power is required and computational resources allow, evaluate **XGBoost** with tuned depth and learning rate.  
   - Use **SVM** or **KNN** only after proper scaling; they can serve as baselines.

4. **Hyper-parameter Search**  
   - Perform **randomized search** or Bayesian optimisation on a limited parameter grid; avoid exhaustive grids for tree ensembles where gains are marginal.

5. **Interpretability & Transparency**  
   - Generate **SHAP summary plots** for the final model; expose top features in the web UI to help users understand predictions [11].

6. **Streamlit Deployment**  
   - Cache heavy data loading (`st.cache_data`) to prevent re-reading the Spotify dataset on every interaction.  
   - Keep model inference lightweight; load the trained model once at startup and reuse across sessions [6].  
   - Use **session_state** sparingly to avoid memory bloat; release large objects after inference.  
   - Provide interactive widgets for users to adjust feature values and view real-time probability outputs, ensuring the UI remains responsive [6].

Following these steps yields a reproducible, performant classifier that can be explored by end-users through a Streamlit interface.

---  

*The survey synthesizes current findings on audio-feature importance, artist influence, classification algorithms, imbalance mitigation, hyper-parameter effects, interpretability, and deployment best practices for K-Pop popularity prediction on Spotify.*

---

### Sources
- [1] https://developer.spotify.com/documentation/web-api/reference/get-audio-features
- [2] https://indjst.org/articles/predicting-music-popularity-using-spotify-and-youtube-features
- [3] https://arxiv.org/html/2505.07280v1
- [4] https://www.scitepress.org/Papers/2024/133300/133300.pdf
- [5] https://nhsjs.com/wp-content/uploads/2024/04/Veritas-AI-Tuning-into-Trends-Machine-Learning-Models-for-Song-Popularity-Prediction-on-Spotify.pdf
- [6] https://medium.com/@ucladsu/exploring-k-pop-and-bts-song-popularity-with-spotify-audio-features-song-lyrics-40284399948f
- [7] https://medium.com/thedeephub/effective-strategies-for-handling-class-imbalance-in-machine-learning-models-ad4a9ee7ff6e
- [8] https://www.mdpi.com/2227-7080/13/3/88
- [9] https://www.mdpi.com/2673-4591/89/1/43
- [10] https://cs229.stanford.edu/proj2015/140_report.pdf
- [11] https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
- [12] https://www.politesi.polimi.it/retrieve/16361764-bbc2-4545-816d-59f7caaa2e98/Elisa_Castelli_Thesis.pdf