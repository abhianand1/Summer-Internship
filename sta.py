import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv(r"C:\Academics\7th Sem\Summer Internship\synthetic_fbg_large_dataset.csv")
X = df.drop(columns=['Label']).values
y = df['Label'].values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
print("Data loaded and preprocessed.")

def calculate_sta_lta(signal, sta_window=1, lta_window=100, threshold=1.028):
    """
    Calculates the STA/LTA ratio for a given signal.
    Returns a boolean array where True indicates a peak detected (ratio > threshold).
    """
    sta = np.convolve(signal, np.ones(sta_window), 'same') / sta_window
    lta = np.convolve(signal, np.ones(lta_window), 'same') / lta_window
    
    sta_lta_ratio = np.divide(sta, lta, out=np.zeros_like(sta), where=lta!=0)
    return sta_lta_ratio > threshold


X_sta_lta = np.array([calculate_sta_lta(sample, 1, 100).astype(float) for sample in X])

X_sta_lta = np.nan_to_num(X_sta_lta, nan=0.0, posinf=0.0, neginf=0.0)

pca = PCA(n_components=min(100, X_sta_lta.shape[1] - 1))
X_sta_lta_reduced = pca.fit_transform(X_sta_lta)

X_train_sta_lta, X_test_sta_lta, y_train, y_test = train_test_split(X_sta_lta_reduced, y, test_size=0.3, random_state=42)
print("STA/LTA preprocessing, NaN removal, and PCA applied.")
lda_sta_lta = LinearDiscriminantAnalysis()
lda_sta_lta.fit(X_train_sta_lta, y_train)
y_pred_lda_sta_lta = lda_sta_lta.predict(X_test_sta_lta)
accuracy_lda_sta_lta = accuracy_score(y_test, y_pred_lda_sta_lta)
print(f"LDA Model Accuracy with STA/LTA Preprocessing and PCA: {accuracy_lda_sta_lta * 100:.2f}%")
log_reg_sta_lta = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
log_reg_sta_lta.fit(X_train_sta_lta, y_train)
y_pred_log_reg_sta_lta = log_reg_sta_lta.predict(X_test_sta_lta)
accuracy_log_reg_sta_lta = accuracy_score(y_test, y_pred_log_reg_sta_lta)
print(f"Logistic Regression Model Accuracy with STA/LTA Preprocessing and PCA: {accuracy_log_reg_sta_lta * 100:.2f}%")