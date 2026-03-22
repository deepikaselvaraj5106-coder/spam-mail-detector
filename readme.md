# 📩 AI Spam Mail Detector

A professional Machine Learning web application that classifies messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a hybrid detection approach.

---

## 🚀 Features

* 🔍 Detects spam messages in real-time
* 🧠 Uses **TF-IDF + Logistic Regression**
* ⚡ Hybrid approach (Machine Learning + Rule-based filtering)
* 📊 Displays **confidence score**
* 🎨 Modern and responsive UI
* 💻 Built with Flask web framework

---

## 🛠️ Tech Stack

* **Python**
* **Flask**
* **Scikit-learn**
* **Pandas**
* **HTML/CSS**

---

## 🧠 How It Works

1. Text is preprocessed (lowercasing, cleaning)
2. Converted into numerical features using **TF-IDF Vectorizer**
3. Classified using **Logistic Regression**
4. Additional **rule-based filtering** improves accuracy for common spam keywords
5. Displays prediction along with confidence score

---

## 📊 Model Performance

* **Accuracy:** ~95%
* Evaluated using:

  * Accuracy Score
  * Precision & Recall
  * Classification Report

---

## 📸 Screenshots

*Add your screenshots here*

* ✅ Not Spam Prediction
* 🚨 Spam Detection
* 📊 Confidence Score Display

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/deepikaselvaraj5106-coder/spam-mail-detector
cd spam-mail-detector
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python train_model.py
```

### 4️⃣ Run the application

```bash
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 🧪 Sample Inputs

### 🚨 Spam

* "Win ₹50000 now!!! Click here"
* "Get cheap loans instantly!!! Apply now"

### ✅ Not Spam

* "Hey, are we meeting today?"
* "Let's go for lunch tomorrow"

---

## 💡 Future Improvements

* 🔹 Deep Learning models (LSTM / BERT)
* 🔹 Email integration
* 🔹 Multi-language support
* 🔹 Deployment on cloud (Render / AWS)

---

## 👩‍💻 Author

**Deepika Selvaraj**
BCA Student | AI & Python Enthusiast

---

## ⭐ Acknowledgement

Dataset: SMS Spam Collection Dataset (UCI Repository)

---

## 📌 Project Highlights

✔ Real-world NLP application
✔ Hybrid ML + Rule-based approach
✔ Clean UI + Flask deployment
✔ Beginner-friendly yet industry-relevant

---

> 🚀 This project demonstrates practical implementation of Machine Learning and NLP in solving real-world problems.
