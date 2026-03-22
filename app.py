from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# You can update this based on your training output
accuracy = "95%"

@app.route('/')
def home():
    return render_template("index.html", accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    msg_lower = message.lower()

    # Rule-based keywords
    spam_keywords = [
        "win", "winner", "free", "cash", "prize",
        "loan", "offer", "urgent", "money",
        "click", "apply now", "buy now"
    ]

    # ML prediction
    data = vectorizer.transform([message])
    result = model.predict(data)

    # Confidence score
    try:
        prob = model.predict_proba(data)[0]
        confidence_score = max(prob) * 100
        confidence = f"{confidence_score:.2f}%"
    except:
        confidence = "N/A"

    # Final decision
    if any(word in msg_lower for word in spam_keywords):
        highlight_words = [word for word in spam_keywords if word in msg_lower]
        prediction = f"🚨 Spam Message (keywords: {', '.join(highlight_words)})"
    else:
        if result[0] == 1:
            prediction = "🚨 Spam Message (ML)"
        else:
            prediction = "✅ Not Spam"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        accuracy=accuracy
    )

if __name__ == "__main__":
    app.run(debug=True)