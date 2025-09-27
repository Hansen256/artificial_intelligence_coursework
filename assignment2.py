#!/usr/bin/env python3
"""
Sentiment Analysis Assignment - Movie Reviews
6. Sentiment Analysis – Movie or Social Media Reviews
Dataset: IMDb Movie Reviews
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2, SelectKBest
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    # Handle cases where NLTK download fails
    pass

def load_and_explore_data(file_path):
    """Load the IMDb dataset and perform basic exploration"""
    try:
        df_main = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        print(f"Shape: {df_main.shape}")
        print(f"Columns: {df_main.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df_main.head())
        print(f"\nSentiment distribution:")
        print(df_main['sentiment'].value_counts())
        return df_main
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please make sure the IMDB Dataset.csv file is in the project directory.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def analyze_review_lengths(data):
    """Phase 1: Analyze review lengths by sentiment"""
    print("\n=== Phase 1: EDA - Review Length Analysis ===")
    
    # Calculate word counts
    data['word_count'] = data['review'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Basic statistics
    print(f"Average review length: {data['word_count'].mean():.2f} words")
    print(f"Median review length: {data['word_count'].median():.2f} words")
    
    # Statistics by sentiment
    sentiment_stats = data.groupby('sentiment')['word_count'].agg(['mean', 'median', 'std'])
    print("\nReview length statistics by sentiment:")
    print(sentiment_stats)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    positive_lengths = data[data['sentiment'] == 'positive']['word_count']
    negative_lengths = data[data['sentiment'] == 'negative']['word_count']
    
    plt.hist([positive_lengths, negative_lengths], bins=50, alpha=0.7, 
             label=['Positive', 'Negative'], color=['green', 'red'])
    plt.xlabel('Review Length (words)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Review Lengths by Sentiment')
    plt.legend()
    
    # Box plot
    plt.subplot(1, 2, 2)
    data.boxplot(column='word_count', by='sentiment', ax=plt.gca())
    plt.title('Review Length Box Plot by Sentiment')
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig('review_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return data

def word_frequency_analysis(data):
    """Analyze word frequency in positive vs negative reviews"""
    print("\n=== Word Frequency Analysis ===")
    
    # Clean the text data
    data['cleaned_review'] = data['review'].apply(clean_text)
    
    # Separate positive and negative reviews
    positive_reviews = data[data['sentiment'] == 'positive']['cleaned_review'].str.cat(sep=' ')
    negative_reviews = data[data['sentiment'] == 'negative']['cleaned_review'].str.cat(sep=' ')
    
    # Get word frequencies
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        # Fallback if stopwords not available
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def get_top_words(text, n=20):
        words = text.split()
        words = [word for word in words if len(word) > 2 and word not in stop_words]
        return Counter(words).most_common(n)
    
    positive_words = get_top_words(positive_reviews, 20)
    negative_words = get_top_words(negative_reviews, 20)
    
    print("Top 10 words in positive reviews:")
    for word, count in positive_words[:10]:
        print(f"  {word}: {count}")
    
    print("\nTop 10 words in negative reviews:")
    for word, count in negative_words[:10]:
        print(f"  {word}: {count}")
    
    # Create word frequency plots
    plt.figure(figsize=(15, 6))
    
    # Positive words plot
    plt.subplot(1, 2, 1)
    pos_words, pos_counts = zip(*positive_words)
    plt.barh(range(len(pos_words)), pos_counts, color='green', alpha=0.7)
    plt.yticks(range(len(pos_words)), pos_words)
    plt.xlabel('Frequency')
    plt.title('Top 20 Words in Positive Reviews')
    plt.gca().invert_yaxis()
    
    # Negative words plot
    plt.subplot(1, 2, 2)
    neg_words, neg_counts = zip(*negative_words)
    plt.barh(range(len(neg_words)), neg_counts, color='red', alpha=0.7)
    plt.yticks(range(len(neg_words)), neg_words)
    plt.xlabel('Frequency')
    plt.title('Top 20 Words in Negative Reviews')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('word_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return data

def dimensionality_reduction_visualization(data, sample_size=5000):
    """Apply PCA, t-SNE visualization on TF-IDF vectors"""
    print("\n=== Dimensionality Reduction Visualization ===")
    
    # Sample data for faster processing
    if len(data) > sample_size:
        sample_data = data.sample(n=sample_size, random_state=42)
        print(f"Using sample of {sample_size} reviews for visualization")
    else:
        sample_data = data.copy()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(sample_data['cleaned_review'])
    y = (sample_data['sentiment'] == 'positive').astype(int)
    
    print(f"TF-IDF matrix shape: {X.shape}")
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    
    # Apply t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X.toarray())
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # PCA plot
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of TF-IDF Vectors')
    
    # t-SNE plot
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='RdYlGn', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of TF-IDF Vectors')
    
    # Explained variance plot for PCA
    plt.subplot(1, 3, 3)
    pca_full = PCA()
    pca_full.fit(X.toarray())
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)[:50]  # First 50 components
    plt.plot(range(1, len(cumsum_var) + 1), cumsum_var)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimensionality_reduction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sample_data, vectorizer

def create_custom_features(data):
    """Phase 2: Create custom features for sentiment analysis"""
    print("\n=== Phase 2: Feature Engineering ===")
    
    # Text statistics features
    data['char_count'] = data['review'].str.len()
    data['sentence_count'] = data['review'].str.count('[.!?]+')
    data['avg_word_length'] = data['review'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    # Punctuation features
    data['exclamation_count'] = data['review'].str.count('!')
    data['question_count'] = data['review'].str.count(r'\?')
    data['capital_ratio'] = data['review'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    # Sentiment lexicon features using TextBlob
    def get_textblob_sentiment(text):
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception:
            return 0, 0
    
    data[['textblob_polarity', 'textblob_subjectivity']] = data['review'].apply(
        lambda x: pd.Series(get_textblob_sentiment(x))
    )
    
    # HTML and formatting features
    data['html_tag_count'] = data['review'].str.count('<.*?>')
    data['uppercase_word_count'] = data['review'].apply(
        lambda x: sum(1 for word in str(x).split() if word.isupper())
    )
    
    print("Custom features created:")
    feature_columns = ['char_count', 'sentence_count', 'avg_word_length', 
                      'exclamation_count', 'question_count', 'capital_ratio',
                      'textblob_polarity', 'textblob_subjectivity', 
                      'html_tag_count', 'uppercase_word_count']
    
    for col in feature_columns:
        print(f"  {col}: mean={data[col].mean():.3f}, std={data[col].std():.3f}")
    
    return data, feature_columns

def feature_selection_analysis(data, custom_features):
    """Apply feature selection using Chi-Square test"""
    print("\n=== Feature Selection Analysis ===")
    
    # Prepare features for selection
    X_custom = data[custom_features].fillna(0)
    y = (data['sentiment'] == 'positive').astype(int)
    
    # Apply Chi-Square feature selection
    # Note: Chi-Square requires non-negative features, so we'll scale them
    X_custom_scaled = X_custom - X_custom.min() + 1e-6  # Make all values positive
    
    # Select top features using Chi-Square
    selector = SelectKBest(score_func=chi2, k=min(5, len(custom_features)))
    X_selected = selector.fit_transform(X_custom_scaled, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'feature': custom_features,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    print("Feature importance (Chi-Square scores):")
    print(feature_scores)
    
    selected_features = [custom_features[i] for i in selector.get_support(indices=True)]
    print(f"\nSelected features: {selected_features}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_scores)), feature_scores['score'])
    plt.yticks(range(len(feature_scores)), feature_scores['feature'])
    plt.xlabel('Chi-Square Score')
    plt.title('Feature Importance (Chi-Square)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return selected_features, feature_scores

def train_and_evaluate_models(data, custom_features):
    """Phase 3: Train and evaluate models"""
    print("\n=== Phase 3: Model Training and Evaluation ===")
    
    # Prepare data
    X_text = data['cleaned_review'].fillna('')
    X_custom = data[custom_features].fillna(0)
    y = (data['sentiment'] == 'positive').astype(int)
    
    # Split data
    X_text_train, X_text_test, X_custom_train, X_custom_test, y_train, y_test = train_test_split(
        X_text, X_custom, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
    X_tfidf_train = tfidf.fit_transform(X_text_train)
    X_tfidf_test = tfidf.transform(X_text_test)
    
    # Combine TF-IDF and custom features
    from scipy.sparse import hstack
    X_combined_train = hstack([X_tfidf_train, X_custom_train.values])
    X_combined_test = hstack([X_tfidf_test, X_custom_test.values])
    
    # Train individual models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    print("\nTraining individual models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_combined_train, y_train)
        y_pred = model.predict(X_combined_test)
        y_pred_proba = model.predict_proba(X_combined_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    # Train ensemble model
    print("Training ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ],
        voting='soft'
    )
    
    ensemble.fit(X_combined_train, y_train)
    y_pred_ensemble = ensemble.predict(X_combined_test)
    y_pred_proba_ensemble = ensemble.predict_proba(X_combined_test)[:, 1]
    
    results['Ensemble'] = {
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble),
        'recall': recall_score(y_test, y_pred_ensemble),
        'f1': f1_score(y_test, y_pred_ensemble),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_ensemble),
        'y_pred': y_pred_ensemble,
        'y_pred_proba': y_pred_proba_ensemble
    }
    
    # Print results
    print("\n=== Model Performance Comparison ===")
    results_df = pd.DataFrame({name: {metric: scores[metric] for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']} 
                              for name, scores in results.items()}).T
    print(results_df.round(4))
    
    return results, y_test, X_combined_test, tfidf

def visualize_results(results, y_test):
    """Create visualizations for model evaluation"""
    print("\n=== Creating Evaluation Visualizations ===")
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} - Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc_score = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def error_analysis(data, results, y_test):
    """Perform error analysis on misclassified reviews"""
    print("\n=== Error Analysis ===")
    
    # Use ensemble model for error analysis
    ensemble_predictions = results['Ensemble']['y_pred']
    
    # Find misclassified examples
    misclassified_mask = (y_test != ensemble_predictions)
    
    if len(data) > len(y_test):
        # Get the test indices (assuming data was split)
        test_indices = data.index[-len(y_test):]
        misclassified_data = data.loc[test_indices[misclassified_mask]]
    else:
        # If data size matches test size, use direct indexing
        misclassified_indices = np.where(misclassified_mask)[0]
        misclassified_data = data.iloc[misclassified_indices]
    
    print(f"Total misclassified: {misclassified_mask.sum()}")
    print(f"Error rate: {misclassified_mask.sum() / len(y_test):.3f}")
    
    # Analyze misclassified examples
    if len(misclassified_data) > 0:
        print("\nSample misclassified reviews:")
        for idx, (_, row) in enumerate(misclassified_data.head(3).iterrows()):
            actual = row['sentiment']
            predicted = 'positive' if ensemble_predictions[misclassified_mask][idx] == 1 else 'negative'
            print(f"\nExample {idx + 1}:")
            print(f"  Actual: {actual}, Predicted: {predicted}")
            print(f"  Review: {row['review'][:200]}...")
            print(f"  Length: {row['word_count']} words")

def main():
    """Main execution function"""
    print("=== Sentiment Analysis Assignment ===")
    print("6. Sentiment Analysis – Movie or Social Media Reviews")
    
    # Load data
    file_path = 'IMDB Dataset.csv'
    df_main = load_and_explore_data(file_path)
    
    if df_main is None:
        print("Cannot proceed without dataset. Please ensure 'IMDB Dataset.csv' is in the project directory.")
        return
    
    # Phase 1: EDA
    df_main = analyze_review_lengths(df_main)
    df_main = word_frequency_analysis(df_main)
    sample_data, vectorizer = dimensionality_reduction_visualization(df_main)
    
    # Phase 2: Feature Engineering
    df_main, custom_features = create_custom_features(df_main)
    selected_features, feature_scores = feature_selection_analysis(df_main, custom_features)
    
    # Phase 3: Model Training and Evaluation
    results, y_test, X_test, tfidf = train_and_evaluate_models(df_main, custom_features)
    
    # Phase 4: Visualization and Error Analysis
    visualize_results(results, y_test)
    error_analysis(df_main, results, y_test)
    
    # Phase 5: Reflection
    print("\n=== Reflection ===")
    print("Limitations of bag-of-words vs ensemble models:")
    print("1. Bag-of-words ignores word order and context")
    print("2. Ensemble methods combine multiple algorithms for better performance")
    print("3. Feature engineering adds domain-specific knowledge")
    print("4. Custom features capture sentiment indicators beyond word frequency")
    
    print("\nGenerated files:")
    print("- review_length_analysis.png")
    print("- word_frequency_analysis.png") 
    print("- dimensionality_reduction_analysis.png")
    print("- feature_importance.png")
    print("- confusion_matrices.png")
    print("- roc_curves.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()