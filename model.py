# train_model.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv(r"S:\CODES\MACHINE_LEARNING\spam.csv")  # use raw string for path
print("Dataset loaded successfully ✅")
print(df.head(3))

# ================================
# 2. Feature Extraction (TF-IDF)
# ================================
tdf = TfidfVectorizer()
x = tdf.fit_transform(df["Message"])
y = df["Category"]

# ================================
# 3. Handle Class Imbalance
# ================================
ros = RandomOverSampler()
x_ros, y_ros = ros.fit_resample(x, y)
print(f"Original shape: {x.shape}, Resampled shape: {x_ros.shape}")

# ================================
# 4. Train-Test Split
# ================================
x_train, x_test, y_train, y_test = train_test_split(
    x_ros, y_ros, test_size=0.15, random_state=42
)

# ================================
# 5. Train Model
# ================================
mnm = MultinomialNB()
mnm.fit(x_train, y_train)
score = mnm.score(x_test, y_test)
print(f"Model Accuracy: {score:.2f}")

# ================================
# 6. Evaluation
# ================================
y_pred = mnm.predict(x_test)
cf = confusion_matrix(y_test, y_pred)

sns.heatmap(cf, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

sns.countplot(x="Category", data=df)
plt.title("Distribution of Spam vs Ham")
plt.show()

# ================================
# 7. Save Model & Vectorizer
# ================================
with open("spam_model.pkl", "wb") as model_file:
    pickle.dump(mnm, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tdf, vec_file)

print("✅ Model and Vectorizer saved as spam_model.pkl & vectorizer.pkl")

# ================================
# 8. Quick Test (CLI Prediction)
# ================================
if __name__ == "__main__":
    inp = input("Enter a message: ")
    output = mnm.predict(tdf.transform([inp]))
    print(f"Prediction: {output[0]}")
