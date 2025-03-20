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
# Add pyngrok for tunneling
from pyngrok import ngrok

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

# def preprocess_text(text):
#     if isinstance(text, str):
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#         tokens = nltk.word_tokenize(text)
#         stop_words = set(stopwords.words('english'))
#         tokens = [token for token in tokens if token not in stop_words]
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(token) for token in tokens]
#         return ' '.join(tokens)
#     else:
#         return ''

# Try to load the model and vectorizer
model_loaded = False
try:
    # Try to load both model and vectorizer
    with open('best_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer_v2.pkl', 'rb') as f:
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
        
        # Add a method to provide probability estimates
        def predict_proba(self, text):
            if not isinstance(text, list):
                text = [text]
            results = []
            for item in text:
                if not isinstance(item, str):
                    item = str(item)
                
                if not item:
                    # No input: equal probabilities
                    results.append([0.33, 0.33, 0.34])
                elif "good" in item.lower():
                    # Positive: higher probability for positive
                    results.append([0.15, 0.75, 0.1])
                elif "bad" in item.lower():
                    # Negative: higher probability for negative
                    results.append([0.75, 0.15, 0.1])
                else:
                    # Neutral: higher probability for neutral
                    results.append([0.2, 0.2, 0.6])
            return results
    
    model = DummyModel()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_status=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    comment = data.get('comment', '')
    
    if not comment:
        return jsonify({'label': 'Please enter a comment', 'confidence': 0})
    
    # Make prediction
    try:
        if model_loaded:
            # Process for real model
            # processed_comment = preprocess_text(comment)
            processed_comment = comment
            vectorized_comment = tfidf.transform([processed_comment])
            
            # Get label
            label = model.predict(vectorized_comment)[0]
            
            # Try to get probability/confidence
            confidence = 0
            try:
                # If the model supports predict_proba
                proba = model.predict_proba(vectorized_comment)[0]
                # Get the confidence of the predicted class
                class_index = model.classes_.tolist().index(label)
                confidence = round(proba[class_index] * 100, 2)
            except Exception as e:
                print(f"Error getting confidence: {e}")
                confidence = "N/A"
        else:
            # Process for dummy model
            label = model.predict(comment)
            
            # Use our custom predict_proba method
            proba = model.predict_proba(comment)[0]
            
            # Map the label to its index position
            if label == "Positive":
                confidence = round(proba[1] * 100, 2)
            elif label == "Negative":
                confidence = round(proba[0] * 100, 2)
            else:  # Neutral
                confidence = round(proba[2] * 100, 2)
        
        return jsonify({'label': label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'label': f'Error: {str(e)}', 'confidence': 0})

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
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding: 2rem 0;
        }
        
        .main-container {
            width: 95%;
            max-width: 1000px;
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 2rem;
        }
        
        .input-container {
            flex: 1;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        .history-container {
            flex: 1;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            max-height: 500px;
            display: flex;
            flex-direction: column;
        }
        
        /* Team members section styles */
        .team-section {
            width: 95%;
            max-width: 1000px;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            margin-top: 1rem;
        }
        
        .team-section h2 {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .team-members {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 2rem;
        }
        
        .team-member {
            width: 180px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .member-image {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background-color: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 1rem;
            overflow: hidden;
        }
        
        .member-image img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .member-image i {
            font-size: 40px;
            color: #999;
        }
        
        .member-name {
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .member-title {
            font-size: 0.85rem;
            color: #666;
        }
        
        /* Existing styles */
        h1, h2 {
            color: var(--secondary-color);
            margin-top: 0;
        }
        
        h1 {
            font-size: 1.8rem;
        }
        
        h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
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

        .prediction-log {
            flex-grow: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .log-entries {
            flex-grow: 1;
            overflow-y: auto;
        }
        
        .log-entry {
            padding: 0.5rem;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        
        .log-comment {
            flex-grow: 1;
            margin-right: 1rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 70%;
        }
        
        .log-label {
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: #f0f4f8;
        }
        
        .log-label.Positive {
            background: #d4edda;
            color: #155724;
        }
        
        .log-label.Negative {
            background: #f8d7da;
            color: #721c24;
        }
        
        .empty-log {
            color: #999;
            font-style: italic;
            text-align: center;
            padding: 1rem;
        }
        
        .spinner {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(74, 111, 165, 0.3);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
            margin-right: 10px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
                width: 95%;
            }
            
            .input-container, .history-container {
                width: 100%;
                padding: 1.5rem;
            }
            
            .history-container {
                max-height: 300px;
            }
            
            .team-members {
                gap: 1rem;
            }
            
            .team-member {
                width: 150px;
            }
            
            .member-image {
                width: 100px;
                height: 100px;
            }
        }

        .confidence-container {
            margin-top: 0.75rem;
            font-size: 0.9rem;
        }
        
        .confidence-label {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .confidence-bar-container {
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            margin-top: 0.5rem;
            overflow: hidden;
        }
        
        .confidence-bar {
            height: 100%;
            background-color: var(--primary-color);
            width: 0%;
            transition: width 0.5s ease-in-out;
        }
    </style>
    <!-- Add Font Awesome for user icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
</head>
<body>
    <div class="main-container">
        <div class="input-container">
            <h1>Comment Classifier</h1>
            
            <div class="form-group">
                <label for="comment">Enter your comment:</label>
                <textarea id="comment" placeholder="Type your comment here..."></textarea>
            </div>
            
            <button id="submit-btn">Classify</button>
            
            <div class="result" id="result" style="display: none;">
                <div class="result-label">Classification:</div>
                <div class="result-value" id="result-value">-</div>
                <div class="confidence-container">
                    <div class="confidence-label">Confidence:</div>
                    <div class="confidence-value" id="confidence-value">-</div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" id="confidence-bar"></div>
                    </div>
                </div>
            </div>
            
            <div class="model-status">
                {% if model_status %}
                    Model loaded successfully from best_model.pkl
                {% else %}
                    Using demo model (best_model.pkl not found)
                {% endif %}
            </div>
        </div>
        
        <div class="history-container">
            <h2>Prediction History</h2>
            <div class="prediction-log">
                <div class="log-entries" id="log-entries">
                    <div class="empty-log">No predictions yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Team Members Section -->
    <div class="team-section">
        <h2>Our Team</h2>
        <div class="team-members">
            <div class="team-member">
                <div class="member-image">
                    <i class="fas fa-user"></i>
                </div>
                <div class="member-name">Team Member 1</div>
                <div class="member-title">Model Development</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <i class="fas fa-user"></i>
                </div>
                <div class="member-name">Team Member 2</div>
                <div class="member-title">Data Collection</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <i class="fas fa-user"></i>
                </div>
                <div class="member-name">Team Member 3</div>
                <div class="member-title">Frontend Development</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <i class="fas fa-user"></i>
                </div>
                <div class="member-name">Team Member 4</div>
                <div class="member-title">Backend Integration</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <i class="fas fa-user"></i>
                </div>
                <div class="member-name">Team Member 5</div>
                <div class="member-title">Testing & Documentation</div>
            </div>
        </div>
    </div>

<script>
        // Store prediction history
        const predictionLog = [];
        
        document.addEventListener('DOMContentLoaded', () => {
            // Loading spinner CSS is now included directly in the style tag
            
            // We don't need to create the prediction log section anymore as it's part of the HTML structure
        });

        document.getElementById('submit-btn').addEventListener('click', async () => {
            const comment = document.getElementById('comment').value.trim();
            const resultDiv = document.getElementById('result');
            const resultValue = document.getElementById('result-value');
            const confidenceValue = document.getElementById('confidence-value');
            const confidenceBar = document.getElementById('confidence-bar');
            
            if (!comment) {
                resultValue.textContent = 'Please enter a comment';
                confidenceValue.textContent = '-';
                confidenceBar.style.width = '0%';
                resultDiv.style.display = 'block';
                return;
            }
            
            // Show loading indicator
            resultValue.innerHTML = '<div class="spinner"></div> Analyzing...';
            confidenceValue.textContent = '-';
            confidenceBar.style.width = '0%';
            resultDiv.style.display = 'block';
            
            try {
                // Start both the API request and a timer
                const [response] = await Promise.all([
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ comment }),
                    }),
                    new Promise(resolve => setTimeout(resolve, 500)) // Minimum 500ms delay
                ]);
                
                const data = await response.json();
                resultValue.textContent = data.label;
                
                // Display confidence if available
                if (data.confidence !== undefined && data.confidence !== 'N/A') {
                    confidenceValue.textContent = `${data.confidence}%`;
                    confidenceBar.style.width = `${data.confidence}%`;
                } else if (data.confidence === 'N/A') {
                    confidenceValue.textContent = 'Not available';
                    confidenceBar.style.width = '0%';
                } else {
                    confidenceValue.textContent = '-';
                    confidenceBar.style.width = '0%';
                }
                
                // Add to prediction log
                addToPredictionLog(comment, data.label, data.confidence);
                
            } catch (error) {
                resultValue.textContent = `Error: ${error.message}`;
                confidenceValue.textContent = '-';
                confidenceBar.style.width = '0%';
            }
        });
        
        function addToPredictionLog(comment, label, confidence) {
            // Store in memory
            predictionLog.unshift({ comment, label, confidence, timestamp: new Date() });
            
            // Limit log size
            if (predictionLog.length > 50) {
                predictionLog.pop();
            }
            
            // Update UI
            const logEntries = document.getElementById('log-entries');
            const emptyLog = logEntries.querySelector('.empty-log');
            if (emptyLog) {
                emptyLog.remove();
            }
            
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            // Format the comment (truncate if needed)
            const truncatedComment = comment.length > 50 
                ? comment.substring(0, 50) + '...' 
                : comment;
            
            // Include confidence if available
            const confidenceDisplay = confidence !== undefined && confidence !== 'N/A' 
                ? ` (${confidence}%)` 
                : '';
            
            entry.innerHTML = `
                <div class="log-comment" title="${comment}">${truncatedComment}</div>
                <div class="log-label ${label}">${label}${confidenceDisplay}</div>
            `;
            
            // Add to the top of the log
            logEntries.insertBefore(entry, logEntries.firstChild);
        }
    </script>
</body>
</html>
        ''')
    
# Set up ngrok tunnel
try:
    # Start ngrok tunnel
    ngrok.kill()
    
    import json
    # Load from config.json (not committed to git)
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                ngrok.set_auth_token(config.get('ngrok_auth_token', ''))
    except Exception as e:
        print(f"Error loading config: {e}")
        
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:5000/\"")
    
    # Update any base URLs or callback URLs for the app
    app.config['BASE_URL'] = public_url
    
    # Just for user's information
    print(" * Running on", public_url)
    print(" * Traffic will be forwarded from your ngrok URL to your local machine")
    print(" * To view your app, open this URL in your browser")
    print(" * Press CTRL+C to quit")
except Exception as e:
    print(f" * Failed to establish ngrok tunnel: {e}")
    print(" * Make sure you have ngrok installed and the authtoken is configured")
    print(" * Falling back to local-only Flask server")

# Start the Flask app
app.run(debug=False)