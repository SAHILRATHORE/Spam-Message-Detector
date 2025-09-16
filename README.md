# 📩 Spam Message Detector

A Machine Learning web app built with **Streamlit** to classify SMS messages as **Spam** or **Ham (Not Spam)**.  
This project demonstrates **NLP techniques** and a **Naïve Bayes classifier** for text classification.  

---

## 🔎 Features
- Preprocesses text with **TF-IDF Vectorizer**  
- Handles class imbalance using **Random OverSampler (imblearn)**  
- Trained with **Multinomial Naïve Bayes**  
- Achieves **99% Accuracy** with high Precision & Recall  
- Interactive **Streamlit web app** for real-time predictions  
- Provides **confidence score** for predictions  

---

## 📂 Project Structure
```
├── train_model.py        # Script to train model & save .pkl files
├── spam_model.pkl        # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── app.py                # Streamlit app
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚡ How to Run Locally
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Spam-Message-Detector.git
   cd Spam-Message-Detector
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app  
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Deployment
You can deploy this project easily on:  
- **Streamlit Cloud** (free & simple)  
- **Render / Heroku** for more control  

---

## 📊 Model Performance
- **Accuracy:** 99%  
- **Precision, Recall, F1-score:** > 0.99  
- Minimal misclassification in the **confusion matrix**  

---

## 🛠️ Tech Stack
- **Python** 🐍  
- **Scikit-learn** (ML Model)  
- **imblearn** (RandomOverSampler)  
- **Streamlit** (Web App)  
- **Seaborn/Matplotlib** (Visualization)  

---

## 📌 Example Usage
💬 *Input:* `"Congratulations! You won a free ticket to Bahamas. Claim now!"`  
✅ *Output:* **Spam** (Confidence: 98.76%)  

---

## 👨‍💻 Author
**Sahil Rathore**  
🌐 [Portfolio](https://sahilportfolio605.netlify.app)  
