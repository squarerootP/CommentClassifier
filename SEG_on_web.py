from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# Specify the directory for NLTK data
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

# Append the directory containing NLTK data to the search path before downloading
nltk.data.path.append(nltk_data_dir)

# Download necessary resources to the specified directory
nltk.download('punkt', download_dir=nltk_data_dir)  # Use punkt instead of punkt_tab
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)

app = Flask(__name__)

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    else:
        return ''

# Try to load the model and vectorizer
model_loaded = False
try:
    # Try to load both model and vectorizer
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    model_loaded = True
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Model or vectorizer file not found. Using a dummy model for demonstration.")
    
    # Dummy vectorizer and model for demonstration
    class DummyModel:
        def predict(self, text):
            if not isinstance(text, str):
                text = str(text)
            if not text:
                return "No input"
            elif "good" in text.lower():
                return "Positive"
            elif "bad" in text.lower():
                return "Negative"
            else:
                return "Neutral"
    
    model = DummyModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    
    if not comment:
        return jsonify({'label': 'Please enter a comment'})
    
    # Make prediction
    try:
        if model_loaded:
            # Process for real model
            processed_comment = preprocess_text(comment)
            vectorized_comment = tfidf.transform([processed_comment])
            label = model.predict(vectorized_comment)[0]
        else:
            # Process for dummy model
            label = model.predict(comment)
        return jsonify({'label': label})
    except Exception as e:
        return jsonify({'label': f'Error: {str(e)}'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comment Classifier</title>
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #166088;
            --background-color: #f5f7fa;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-radius: 8px;
            --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        
        .container {
            width: 90%;
            max-width: 600px;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        h1 {
            color: var(--secondary-color);
            margin-top: 0;
            font-size: 1.8rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        textarea {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: var(--border-radius);
            font-family: inherit;
            font-size: 1rem;
            resize: vertical;
            min-height: 100px;
            box-sizing: border-box;
        }
        
        button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: var(--border-radius);
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: var(--secondary-color);
        }
        
        .result {
            margin-top: 1.5rem;
            padding: 1rem;
            background-color: #f0f4f8;
            border-radius: var(--border-radius);
            border-left: 4px solid var(--primary-color);
        }
        
        .result-label {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .result-value {
            font-size: 1.2rem;
            word-break: break-word;
        }
        
        .model-status {
            margin-top: 1.5rem;
            font-size: 0.9rem;
            color: #666;
        }
        
        @media (max-width: 768px) {
            .container {
                width: 95%;
                padding: 1.5rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comment Classifier</h1>
        
        <div class="form-group">
            <label for="comment">Enter your comment:</label>
            <textarea id="comment" placeholder="Type your comment here..."></textarea>
        </div>
        
        <button id="submit-btn">Classify</button>
        
        <div class="result" id="result" style="display: none;">
            <div class="result-label">Classification:</div>
            <div class="result-value" id="result-value">-</div>
        </div>
        
        <div class="model-status">
            {% if model_status %}
                Model loaded successfully from best_model.pkl
            {% else %}
                Using demo model (best_model.pkl not found)
            {% endif %}
        </div>
    </div>

    <script>
        document.getElementById('submit-btn').addEventListener('click', async () => {
            const comment = document.getElementById('comment').value.trim();
            const resultDiv = document.getElementById('result');
            const resultValue = document.getElementById('result-value');
            
            if (!comment) {
                resultValue.textContent = 'Please enter a comment';
                resultDiv.style.display = 'block';
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ comment }),
                });
                
                const data = await response.json();
                resultValue.textContent = data.label;
                resultDiv.style.display = 'block';
            } catch (error) {
                resultValue.textContent = `Error: ${error.message}`;
                resultDiv.style.display = 'block';
            }
        });
    </script>
</body>
</html>
        ''')
    
app.run(debug=True)