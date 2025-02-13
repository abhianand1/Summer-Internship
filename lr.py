from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"C:\Academics\7th Sem\Summer Internship\synthetic_fbg_large_dataset.csv")
X = df.drop(columns=['Label']).values
y = df['Label'].values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
print("Data loaded and preprocessed.")
log_reg = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
report_log_reg = classification_report(y_test, y_pred_log_reg, target_names=["No Intruder", "Intruder", "Wind"])
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
print(f"Accuracy of Logistic Regression Model: {accuracy_log_reg * 100:.2f}%")
print("\nClassification Report:\n", report_log_reg)
print("\nConfusion Matrix:\n", conf_matrix_log_reg)
