# Vietnamese Product Review Sentiment Classification

## Overview
This project implements a web-based application for Vietnamese product review sentiment analysis, specifically focused on comments collected from the Shopee e-commerce platform. The system uses natural language processing and machine learning techniques to classify reviews as positive or negative, providing valuable insights for both businesses and consumers.

## Features
- Vietnamese text preprocessing using Underthesea package
- Text vectorization using TF-IDF
- Sentiment classification via Support Vector Machine (SVM)
- Web-based user interface for real-time sentiment analysis
- Deployed via Ngrok for accessibility

## Technologies Used
- **Python 3.x**: Core programming language
- **Underthesea**: Vietnamese NLP toolkit for text preprocessing
- **Scikit-learn**: Machine learning library for TF-IDF vectorization and classification models
- **Flask**: Web framework for the application interface
- **Ngrok**: Tool for exposing local web server to the internet
- **Playwright**: Browser automation library for web scraping

## Installation

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Setup
1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Download and install Ngrok if you want to expose your application to the internet

## Usage
1. Run the Flask application:
   ```
   python SEG_on_web.py
   ```
2. Access the web interface at `http://localhost:5000` (or via your Ngrok URL if deployed)
3. Enter a Vietnamese product review in the input field
4. Click "Analyze" to get the sentiment classification result

## Project Structure
```
.
├── images/                  # Team member images
├── shopee_crawler/          # Code for crawling Shopee comments
├── templates/
│   └── index.html           # Web application layout
├── best_mode_v{x}/          # Different model versions with their vectorizers
├── SEG_on_web.py            # Main Flask application
└── README.md                # Project documentation
```

## Data Collection & Processing
- Automated web scraping of Shopee product reviews using Playwright
- Data collection across multiple product categories
- Manual review and classification of comments into positive and negative categories
- Dataset balancing to ensure equal representation of sentiment classes

## Model Development
- Text preprocessing: lowercasing, normalization, tokenization, stopword removal
- Feature engineering using TF-IDF vectorization
- Model selection through GridSearch evaluation
- SVM achieved the best performance with 90% accuracy after dataset refinement

## Future Development
- Topic extraction feature to identify key concerns in reviews
- Enhanced data collection to overcome anti-bot mechanisms
- Exploration of transformer-based models for better semantic understanding

## Screenshots
<!-- Add screenshots of your application below. Replace the placeholder text with actual images. -->

### Application Interface
[Add application interface screenshot here]

### Results Visualization
[Add results visualization screenshot here]

## Team Members
- Nguyen Van Phong (FPT University, Hue)
- Dao Anh Khoa (FPT University, Dong Thap)
- Huynh Anh Phuong (FPT University, An Giang)
- Tran Trung Nhan (FPT University, Ca Mau)
- Huynh Ngoc Nhu Quynh (FPT University, Soc Trang)
