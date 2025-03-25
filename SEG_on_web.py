from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
import os
# Add pyngrok for tunneling
from pyngrok import ngrok
from underthesea import word_tokenize, text_normalize
# Additional imports for topic extraction and visualization
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import Counter
from wordcloud import WordCloud

app = Flask(__name__, static_url_path='', static_folder='.')

def preprocess_text(text):
  """Preprocesses Vietnamese text using underthesea.

  Args:
    text: The input Vietnamese text.

  Returns:
    The preprocessed text.
  """

  # 1. Text Normalization:
  text = text.lower()
  text = text_normalize(text)  

  # 2. Word Segmentation (Tokenization):
  tokens = word_tokenize(text)  

  # 3. (Optional) Remove Stopwords, punctuation, special characters:
  # You may need to define a list of stopwords based on your needs and remove them here
  stop_words = [
    "tôi", "bạn", "chúng tôi", "họ", "nó", "ông", "bà", "cô", "chúng ta", "hắn", "mình",
    "ở", "trong", "ngoài", "trên", "dưới", "với", "đến", "từ",
    "và", "nhưng", "hoặc", "nếu", "vì", "nên", "rồi", "mà", "khi", "sau khi",
    "rất", "cũng", "chỉ", "đã", "đang", "sẽ", "nữa", "mới", "lại", "thế",
    "à", "ơi", "nhé", "hả", "không", "có", "phải", "vậy", "thôi", "được"
]
  tokens = [word for word in tokens if word not in stop_words]

  # 4. Join tokens back into a string:
  preprocessed_text = " ".join(tokens) 

  return preprocessed_text

# Topic extraction function
def extract_topics(comments_list, top_n=5):
    """Extract top topics/keywords from a list of comments."""
    if not comments_list:
        return []
        
    # Create a TF-IDF vectorizer
    topic_vectorizer = TfidfVectorizer(
        max_df=0.95,      # Ignore terms that appear in >95% of documents
        min_df=2,         # Ignore terms that appear in fewer than 2 documents
        max_features=200, # Only consider the top 200 features
        stop_words=stop_words
    )
    
    # Fit and transform the comments
    try:
        tfidf_matrix = topic_vectorizer.fit_transform(comments_list)
        feature_names = topic_vectorizer.get_feature_names_out()
        
        # Sum up the TF-IDF scores for each term across all documents
        tfidf_sums = tfidf_matrix.sum(axis=0).A1
        
        # Get the top N terms with highest TF-IDF scores
        top_indices = tfidf_sums.argsort()[-top_n:][::-1]
        top_terms = [(feature_names[i], tfidf_sums[i]) for i in top_indices]
        
        return top_terms
    except:
        # Fallback to simple word frequency if TF-IDF fails
        all_text = " ".join(comments_list)
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_counts = Counter(words)
        # Filter out very common words
        for word in stop_words:
            word_counts.pop(word, None)
        return word_counts.most_common(top_n)

# Generate word cloud image
def generate_wordcloud(text_list, title="Word Cloud"):
    if not text_list:
        return None
    
    # Join all text
    text = " ".join(text_list)
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        contour_width=3
    ).generate(text)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad=0)
    
    # Save to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert BytesIO to base64 string for HTML embedding
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# Create bar chart for top topics
def create_topic_barchart(topics, title="Top Topics"):
    if not topics:
        return None
    
    # Extract terms and scores
    terms, scores = zip(*topics) if topics else ([], [])
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(terms))
    ax.barh(y_pos, scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Score')
    ax.set_title(title)
    plt.tight_layout()
    
    # Save to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Convert to base64
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# Define stop words list globally
stop_words = [
    "tôi", "bạn", "chúng tôi", "họ", "nó", "ông", "bà", "cô", "chúng ta", "hắn", "mình",
    "ở", "trong", "ngoài", "trên", "dưới", "với", "đến", "từ",
    "và", "nhưng", "hoặc", "nếu", "vì", "nên", "rồi", "mà", "khi", "sau khi",
    "rất", "cũng", "chỉ", "đã", "đang", "sẽ", "nữa", "mới", "lại", "thế",
    "à", "ơi", "nhé", "hả", "không", "có", "phải", "vậy", "thôi", "được"
]

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
        def __init__(self):
            # Define the class labels and their indices for consistent mapping
            # Only two classes - Negative and Positive
            self.classes_ = ["Negative", "Positive"]
        
        def predict(self, text):
            if not isinstance(text, str):
                text = str(text)
            if not text:
                return "No input"
            elif "good" in text.lower():
                return "Positive"
            else:
                # Default to Negative for other cases
                return "Negative"
        
        # Add a method to provide probability estimates
        def predict_proba(self, text):
            if not isinstance(text, list):
                text = [text]
            results = []
            for item in text:
                if not isinstance(item, str):
                    item = str(item)
                
                # Create probabilities for two classes: [Negative, Positive]
                if not item:
                    # No input: equal probabilities
                    results.append([0.5, 0.5])  # [Negative, Positive]
                elif "good" in item.lower():
                    # Positive: higher probability for Positive (index 1)
                    results.append([0.2, 0.8])   # [Negative, Positive]
                else:
                    # Negative: higher probability for Negative (index 0)
                    results.append([0.8, 0.2])   # [Negative, Positive]
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
        # Process the comment
        processed_comment = preprocess_text(comment)
        
        if model_loaded:
            # Process for real model
            vectorized_comment = tfidf.transform([processed_comment])
            
            # Get label
            label = model.predict(vectorized_comment)[0]
            
            # Try to get probability/confidence
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
            # Use raw comment for dummy model as it's designed for English keywords
            label = model.predict(comment)
            
            # Use consistent approach for confidence calculation
            proba = model.predict_proba(comment)[0]
            # There are only two classes now: "Negative" (0) and "Positive" (1)
            class_index = 0 if label == "Negative" else 1
            confidence = round(proba[class_index] * 100, 2)
        
        return jsonify({'label': label, 'confidence': confidence})
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'label': f'Error: {str(e)}', 'confidence': 0})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    data = request.get_json()
    batch_text = data.get('batch_text', '')
    
    if not batch_text:
        return jsonify({
            'error': 'Please enter comments',
            'positive_count': 0,
            'negative_count': 0
        })
    
    # Split comments using the separator
    comments = re.split(r'\s*&&&&\s*', batch_text.strip())
    comments = [comment for comment in comments if comment.strip()]
    
    if not comments:
        return jsonify({
            'error': 'No valid comments found',
            'positive_count': 0,
            'negative_count': 0
        })
    
    positive_comments = []
    negative_comments = []
    results = []
    
    # Process each comment
    for comment in comments:
        try:
            processed_comment = preprocess_text(comment)
            
            if model_loaded:
                # Real model
                vectorized_comment = tfidf.transform([processed_comment])
                label = model.predict(vectorized_comment)[0]
                
                try:
                    # Get confidence if available
                    proba = model.predict_proba(vectorized_comment)[0]
                    class_index = model.classes_.tolist().index(label)
                    confidence = round(proba[class_index] * 100, 2)
                except:
                    confidence = "N/A"
            else:
                # Dummy model
                label = model.predict(comment)
                proba = model.predict_proba(comment)[0]
                class_index = 0 if label == "Negative" else 1
                confidence = round(proba[class_index] * 100, 2)
                
            # Add to appropriate list
            if label == "Positive":
                positive_comments.append(comment)
            else:
                negative_comments.append(comment)
                
            # Add to results
            results.append({
                'comment': comment,
                'label': label,
                'confidence': confidence
            })
                
        except Exception as e:
            print(f"Error processing comment: {e}")
            # Skip problematic comments
            continue
            
    # Extract topics from positive and negative comments
    positive_topics = extract_topics(positive_comments)
    negative_topics = extract_topics(negative_comments)
    
    # Generate visualizations
    positive_wordcloud = generate_wordcloud(positive_comments, "Positive Comments Word Cloud")
    negative_wordcloud = generate_wordcloud(negative_comments, "Negative Comments Word Cloud")
    
    positive_barchart = create_topic_barchart(positive_topics, "Top Topics in Positive Comments")
    negative_barchart = create_topic_barchart(negative_topics, "Top Topics in Negative Comments")
    
    return jsonify({
        'results': results,
        'positive_count': len(positive_comments),
        'negative_count': len(negative_comments),
        'positive_topics': [{'term': term, 'score': float(score)} for term, score in positive_topics],
        'negative_topics': [{'term': term, 'score': float(score)} for term, score in negative_topics],
        'positive_wordcloud': positive_wordcloud,
        'negative_wordcloud': negative_wordcloud,
        'positive_barchart': positive_barchart,
        'negative_barchart': negative_barchart
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
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
        
        /* Batch processing container */
        .batch-container {
            width: 95%;
            max-width: 1000px;
            margin: 2rem 0;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        /* Visualization container */
        .visualization-container {
            width: 95%;
            max-width: 1000px;
            margin-bottom: 2rem;
            padding: 2rem;
            background-color: var(--card-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
            display: none;
        }
        
        .viz-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .sentiment-counts {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .count-card {
            flex: 1;
            padding: 1rem;
            border-radius: var(--border-radius);
            text-align: center;
            font-weight: bold;
        }
        
        .positive-count {
            background-color: rgba(40, 167, 69, 0.2);
        }
        
        .negative-count {
            background-color: rgba(220, 53, 69, 0.2);
        }
        
        .viz-row {
            display: flex;
            flex-direction: row;
            gap: 20px;
            margin-bottom: 2rem;
        }
        
        .viz-column {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .viz-card {
            padding: 1rem;
            background-color: var(--background-color);
            border-radius: var(--border-radius);
            box-shadow: var(--box-shadow);
        }
        
        .viz-card h3 {
            margin-top: 0;
            color: var(--secondary-color);
        }
        
        .viz-image {
            width: 100%;
            height: auto;
            border-radius: var(--border-radius);
        }
        
        .topic-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        
        .topic-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }
        
        .topic-term {
            font-weight: 600;
        }
        
        .topic-score {
            color: #666;
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
            padding: 0.75rem;
            border-bottom: 1px solid #eee;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .log-comment {
            flex-grow: 1;
            word-break: break-word;
            line-height: 1.4;
            color: var(--text-color);
        }
        
        .log-result {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .log-timestamp {
            font-size: 0.8rem;
            color: #888;
        }
        
        .log-label {
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: #f0f4f8;
            margin-left: auto;
        }
        
        .log-label.Positive {
            background: #d4edda;
            color: #155724;
        }
        
        .log-label.Negative {
            background: #f8d7da;
            color: #721c24;
        }
        
        /* Remove the neutral styling since we only have 2 classes */
        
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
            .main-container, .viz-row {
                flex-direction: column;
                width: 95%;
            }
            
            .input-container, .history-container, .viz-column {
                width: 100%;
                padding: 1.5rem;
            }
            
            .history-container {
                max-height: 300px;
            }
        }
        
        /* Existing styles */
        /* ...existing code... */
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
    
    <!-- New Batch Processing Container -->
    <div class="batch-container">
        <h2>Batch Comment Analysis</h2>
        <p>Enter multiple comments separated by "&&&&" for batch processing:</p>
        
        <div class="form-group">
            <textarea id="batch-comments" rows="6" placeholder="Comment 1 &&&& Comment 2 &&&& Comment 3..."></textarea>
        </div>
        
        <button id="batch-submit-btn">Process Batch</button>
        <div id="batch-loading" style="display: none; margin-top: 1rem;">
            <div class="spinner"></div> Processing comments...
        </div>
    </div>
    
    <!-- Visualization Container (initially hidden) -->
    <div class="visualization-container" id="viz-container">
        <div class="viz-header">
            <h2>Batch Analysis Results</h2>
            <div class="tabs">
                <div class="tab active" data-tab="overview">Overview</div>
                <div class="tab" data-tab="wordclouds">Word Clouds</div>
                <div class="tab" data-tab="topics">Topic Charts</div>
            </div>
        </div>
        
        <!-- Overview Tab -->
        <div class="tab-content active" id="overview-tab">
            <div class="sentiment-counts">
                <div class="count-card positive-count">
                    <div>Positive Comments</div>
                    <div id="positive-count">0</div>
                </div>
                <div class="count-card negative-count">
                    <div>Negative Comments</div>
                    <div id="negative-count">0</div>
                </div>
            </div>
            
            <div class="viz-row">
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Top Topics in Positive Comments</h3>
                        <ul class="topic-list" id="positive-topics">
                            <li class="empty-list">No data available</li>
                        </ul>
                    </div>
                </div>
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Top Topics in Negative Comments</h3>
                        <ul class="topic-list" id="negative-topics">
                            <li class="empty-list">No data available</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Word Clouds Tab -->
        <div class="tab-content" id="wordclouds-tab">
            <div class="viz-row">
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Positive Comments Word Cloud</h3>
                        <div class="viz-image-container">
                            <img id="positive-wordcloud" class="viz-image" src="" alt="Positive Word Cloud">
                        </div>
                    </div>
                </div>
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Negative Comments Word Cloud</h3>
                        <div class="viz-image-container">
                            <img id="negative-wordcloud" class="viz-image" src="" alt="Negative Word Cloud">
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Topic Charts Tab -->
        <div class="tab-content" id="topics-tab">
            <div class="viz-row">
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Top Topics in Positive Comments</h3>
                        <div class="viz-image-container">
                            <img id="positive-barchart" class="viz-image" src="" alt="Positive Topics Chart">
                        </div>
                    </div>
                </div>
                <div class="viz-column">
                    <div class="viz-card">
                        <h3>Top Topics in Negative Comments</h3>
                        <div class="viz-image-container">
                            <img id="negative-barchart" class="viz-image" src="" alt="Negative Topics Chart">
                        </div>
                    </div>
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
                    <img src="/images/team1.jpg" alt="Nguyen Van Phong">
                </div>
                <div class="member-name">Nguyen Van Phong</div>
                <div class="member-title">Model Training</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <img src="/images/team2.jpg" alt="Tran Trung Nhan">
                </div>
                <div class="member-name">Tran Trung Nhan</div>
                <div class="member-title">Data Collection and Labeling</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <img src="/images/team3.jpg" alt="Huynh Ngoc Nhu Quynh">
                </div>
                <div class="member-name">Huynh Ngoc Nhu Quynh</div>
                <div class="member-title">Data Collection and Labeling</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <img src="/images/team4.jpg" alt="Huynh Anh Phuong">
                </div>
                <div class="member-name">Huynh Anh Phuong</div>
                <div class="member-title">Data Collection and Labeling</div>
            </div>
            
            <div class="team-member">
                <div class="member-image">
                    <img src="/images/team5.jpg" alt="Dao Anh Khoa">
                </div>
                <div class="member-name">Dao Anh Khoa</div>
                <div class="member-title">Data Collection and Labeling</div>
            </div>
        </div>
    </div>

<script>
        // Store prediction history
        const predictionLog = [];
        
        document.addEventListener('DOMContentLoaded', () => {
            // Set up tab functionality
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Remove active class from all tabs and contents
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    // Add active class to clicked tab
                    tab.classList.add('active');
                    
                    // Show corresponding content
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(`${tabId}-tab`).classList.add('active');
                });
            });
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
            const timestamp = new Date();
            predictionLog.unshift({ comment, label, confidence, timestamp });
            
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
            
            // Format timestamp
            const timeString = timestamp.toLocaleTimeString();
            
            // Include confidence if available
            const confidenceDisplay = confidence !== undefined && confidence !== 'N/A' 
                ? ` (${confidence}%)` 
                : '';
            
            entry.innerHTML = `
                <div class="log-comment">${comment}</div>
                <div class="log-result">
                    <span class="log-timestamp">${timeString}</span>
                    <span class="log-label ${label}">${label}${confidenceDisplay}</span>
                </div>
            `;
            
            // Add to the top of the log
            logEntries.insertBefore(entry, logEntries.firstChild);
        }
        
        // Batch processing functionality
        document.getElementById('batch-submit-btn').addEventListener('click', async () => {
            const batchText = document.getElementById('batch-comments').value.trim();
            const loadingElement = document.getElementById('batch-loading');
            const vizContainer = document.getElementById('viz-container');
            
            if (!batchText) {
                alert('Please enter comments for batch processing');
                return;
            }
            
            // Show loading indicator
            loadingElement.style.display = 'block';
            
            try {
                const response = await fetch('/batch_predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ batch_text: batchText }),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert(data.error);
                    loadingElement.style.display = 'none';
                    return;
                }
                
                // Update UI with results
                document.getElementById('positive-count').textContent = data.positive_count;
                document.getElementById('negative-count').textContent = data.negative_count;
                
                // Update topics lists
                updateTopicsList('positive-topics', data.positive_topics);
                updateTopicsList('negative-topics', data.negative_topics);
                
                // Update images
                if (data.positive_wordcloud) {
                    document.getElementById('positive-wordcloud').src = `data:image/png;base64,${data.positive_wordcloud}`;
                }
                if (data.negative_wordcloud) {
                    document.getElementById('negative-wordcloud').src = `data:image/png;base64,${data.negative_wordcloud}`;
                }
                if (data.positive_barchart) {
                    document.getElementById('positive-barchart').src = `data:image/png;base64,${data.positive_barchart}`;
                }
                if (data.negative_barchart) {
                    document.getElementById('negative-barchart').src = `data:image/png;base64,${data.negative_barchart}`;
                }
                
                // Show visualization container
                vizContainer.style.display = 'block';
                
                // Hide loading indicator
                loadingElement.style.display = 'none';
                
                // Scroll to visualization results
                vizContainer.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                console.error('Error processing batch:', error);
                alert('Error processing batch: ' + error.message);
                loadingElement.style.display = 'none';
            }
        });
        
        function updateTopicsList(elementId, topics) {
            const topicsList = document.getElementById(elementId);
            
            // Clear existing content
            topicsList.innerHTML = '';
            
            // If no topics, show empty message
            if (!topics || topics.length === 0) {
                topicsList.innerHTML = '<li class="empty-list">No data available</li>';
                return;
            }
            
            // Add each topic to the list
            topics.forEach(topic => {
                const topicItem = document.createElement('li');
                topicItem.className = 'topic-item';
                topicItem.innerHTML = `
                    <span class="topic-term">${topic.term}</span>
                    <span class="topic-score">${topic.score.toFixed(4)}</span>
                `;
                topicsList.appendChild(topicItem);
            });
        }
    </script>
</body>
</html>
        ''')

    # Set up ngrok tunnel for external access
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")

    # Run the Flask app
    app.run(debug=True)