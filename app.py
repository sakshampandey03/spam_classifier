from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    result = 'Spam' if prediction[0] == 1 else 'Ham'
    return render_template('index.html', result=result, text=text)

if __name__ == "__main__":
    app.run(debug=True)
