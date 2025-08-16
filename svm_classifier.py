# svm_classifier.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load preprocessed data
print("[INFO] Loading preprocessed data...")
data = np.load("cat_dog_features.npz")
X, y = data["X"], data["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("[INFO] Training SVM...")
model = SVC(kernel="linear", random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Cat", "Dog"])
cm = confusion_matrix(y_test, y_pred)

# Print to terminal
print("[RESULT] Accuracy:", acc)
print("[RESULT] Classification Report:\n", report)
print("[RESULT] Confusion Matrix:\n", cm)

# ---- Save Results ----
with open("results_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

# Save confusion matrix as PNG
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("[INFO] Results saved: results_report.txt & confusion_matrix.png")
