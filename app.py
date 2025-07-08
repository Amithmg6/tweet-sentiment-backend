# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import pickle
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

# Download NLTK data if not already present in the Heroku/Render environment
# This is crucial for the preprocessing functions to work.
# Render's build process should handle this if these lines are present.

nltk.download('stopwords')
nltk.download('twitter_samples')
nltk.download('punkt')


app = Flask(__name__)
CORS(app) # Enable CORS for all routes. This is important for your frontend on GitHub Pages to access this API.

# --- Preprocessing and Model Functions (Copied from your sentiment_nb_tweet.py) ---

def process_tweet(tweet):
    """
    Processes a single tweet by removing noise, tokenizing, removing stopwords, and stemming.
    This function MUST be identical to the one used during model training.
    """
    stemmer = nltk.SnowballStemmer("english")
    stopwords_english = stopwords.words('english')

    # Remove stock market tickers (e.g., $AAPL)
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove hyperlinks (http:// or https://)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove hashtags (only the # symbol, keeping the word)
    tweet = re.sub(r'#', '', tweet)

    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def sigmoid(z):
    """
    Sigmoid function used in logistic regression.
    Input: z (can be a scalar or an array)
    Output: h (the sigmoid of z)
    """
    zz = np.negative(z)
    h = 1 / (1 + np.exp(zz))
    return h

def extract_features(tweet, freqs):
    """
    Extracts features for a single tweet based on word frequencies.
    Input:
        tweet: a string tweet
        freqs: a dictionary mapping (word, sentiment_label) to its frequency
    Output:
        x: a feature vector of dimension (1,3) for the tweet
           x[0,0] is bias (1)
           x[0,1] is sum of positive word counts
           x[0,2] is sum of negative word counts
    """
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    for word in word_l:
        # increment the word count for the positive label 1.0
        x[0, 1] += freqs.get((word, 1.0), 0)
        # increment the word count for the negative label 0.0
        x[0, 2] += freqs.get((word, 0.0), 0)

    assert (x.shape == (1, 3))
    return x

def predict_tweet(tweet, freqs, theta):
    """
    Predicts the probability of a tweet being positive.
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights from the trained model
    Output:
        y_pred: the probability of a tweet being positive (float between 0 and 1)
    """
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

# --- Load the trained model (theta) and frequencies (freqs) ---
# These files must be present in the 'models/' directory relative to app.py
model_path = 'models/logistic_regression_model.pkl'
freqs_path = 'models/tweet_frequencies.pkl'

theta = None
freqs = None

try:
    with open(model_path, 'rb') as f:
        theta = pickle.load(f)
    with open(freqs_path, 'rb') as f:
        freqs = pickle.load(f)
    print("Model (theta) and frequencies (freqs) loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model or frequencies .pkl files not found at {model_path} or {freqs_path}. "
          "Please ensure they are in the 'models/' directory.")
except Exception as e:
    print(f"Error loading model or frequencies: {e}")


# --- Flask Routes ---

# Removed the home route that tried to render index.html
# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive a tweet, classify its sentiment, and return the result.
    """
    if theta is None or freqs is None:
        return jsonify({'error': 'Machine learning model or frequencies not loaded on server. Server might be misconfigured.'}), 500

    data = request.get_json()
    tweet_text = data.get('tweet')

    if not tweet_text:
        return jsonify({'error': 'No tweet text provided.'}), 400

    try:
        # Get the prediction probability
        y_pred = predict_tweet(tweet_text, freqs, theta)

        # Map the probability to sentiment label
        # Based on your original script's test_logistic_regression:
        # y_pred > 0.5 -> Positive (1)
        # y_pred <= 0.5 -> Negative (0)
        # To include 'Neutral', we'll define a small range around 0.5
        sentiment_label = "Neutral" # Default to Neutral if close to 0.5
        if y_pred > 0.55: # Clearly positive
            sentiment_label = "Positive"
        elif y_pred < 0.45: # Clearly negative
            sentiment_label = "Negative"
        # Otherwise, it falls between 0.45 and 0.55, considered Neutral

        return jsonify({'sentiment': sentiment_label, 'probability': float(y_pred)})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed due to an internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
