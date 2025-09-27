"""
Sentiment Analysis - Movie Reviews Assignment
Dataset: IMDb Movie Reviews (50K reviews)

This script implements a comprehensive sentiment analysis pipeline including:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering & Reduction
3. Text Preprocessing
4. Model Training & Ensemble
5. Evaluation & Error Analysis
6. Reflection on Results

"""

# Import required libraries
import pandas as pd
import numpy as np

# Set matplotlib backend to non-interactive before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from collections import Counter
import re
    # string is not used

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries with error handling
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available, using matplotlib defaults")

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("WordCloud not available, skipping word cloud generation")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("BeautifulSoup not available, will use regex for HTML cleaning")

try:
    import nltk
    from nltk.corpus import stopwords
    # from nltk.tokenize import word_tokenize  # unused
    # from nltk.stem import PorterStemmer      # unused
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("NLTK not available, using basic text processing")

# Machine Learning imports - these are essential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import accuracy_score, f1_score

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, will skip UMAP visualization")

# Feature selection
from sklearn.feature_selection import chi2, SelectKBest, mutual_info_classif

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("TextBlob not available, will use alternative sentiment scoring")

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("SENTIMENT ANALYSIS - IMDb MOVIE REVIEWS")
print("="*80)

# ===============================================================================
# PHASE 1: DATA LOADING & INITIAL EXPLORATION
# ===============================================================================

def load_and_explore_data():
    """Load the dataset and perform initial exploration"""
    print("\n" + "="*50)
    print("PHASE 1: DATA LOADING & EXPLORATION")
    print("="*50)
    
    # Load dataset
    print("Loading IMDb dataset...")
    df_local = pd.read_csv('IMDB Dataset.csv')
    
    print(f"Dataset shape: {df_local.shape}")
    print(f"Columns: {df_local.columns.tolist()}")
    print("\nSentiment distribution:")
    print(df_local['sentiment'].value_counts())
    print("\nPercentage distribution:")
    print(df_local['sentiment'].value_counts(normalize=True) * 100)
    # Check for missing values
    print("\nMissing values:")
    print(df_local.isnull().sum())
    # Display sample reviews
    print("\nSample positive review:")
    pos_sample = df_local[df_local['sentiment'] == 'positive']['review'].iloc[0]
    print(pos_sample[:500] + "..." if len(pos_sample) > 500 else pos_sample)
    print("\nSample negative review:")
    neg_sample = df_local[df_local['sentiment'] == 'negative']['review'].iloc[0]
    print(neg_sample[:500] + "..." if len(neg_sample) > 500 else neg_sample)
    return df_local

# ===============================================================================
# PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)
# ===============================================================================

def calculate_review_lengths(df_in):
    """Calculate and analyze review lengths"""
    print("\n" + "-"*40)
    print("1. REVIEW LENGTH ANALYSIS")
    print("-"*40)
    
    # Calculate word counts
    df_in['word_count'] = df_in['review'].apply(lambda x: len(str(x).split()))
    df_in['char_count'] = df_in['review'].apply(len)
    # Overall statistics
    print("Overall Review Length Statistics:")
    print(f"Average word count: {df_in['word_count'].mean():.2f}")
    print(f"Median word count: {df_in['word_count'].median():.2f}")
    print(f"Min word count: {df_in['word_count'].min()}")
    print(f"Max word count: {df_in['word_count'].max()}")
    # By sentiment
    print("\nBy Sentiment:")
    sentiment_stats = df_in.groupby('sentiment')['word_count'].agg(['mean', 'median', 'std', 'min', 'max'])
    print(sentiment_stats)
    return df_in

def plot_review_length_histograms(df_in):
    """Create histograms of review lengths by sentiment"""
    print("\n" + "-"*40)
    print("2. REVIEW LENGTH HISTOGRAMS")
    print("-"*40)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Review Length Analysis by Sentiment', fontsize=16)
    
    # Overall distribution
    axes[0,0].hist(df_in['word_count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(df_in['word_count'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df_in["word_count"].mean():.0f}')
    axes[0,0].set_title('Overall Word Count Distribution')
    axes[0,0].set_xlabel('Word Count')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    # By sentiment - overlapped
    pos_reviews = df_in[df_in['sentiment'] == 'positive']['word_count']
    neg_reviews = df_in[df_in['sentiment'] == 'negative']['word_count']
    
    axes[0,1].hist(pos_reviews, bins=50, alpha=0.6, label='Positive', color='green')
    axes[0,1].hist(neg_reviews, bins=50, alpha=0.6, label='Negative', color='red')
    axes[0,1].axvline(pos_reviews.mean(), color='darkgreen', linestyle='--')
    axes[0,1].axvline(neg_reviews.mean(), color='darkred', linestyle='--')
    axes[0,1].set_title('Word Count Distribution by Sentiment')
    axes[0,1].set_xlabel('Word Count')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # Box plots
    df_in.boxplot(column='word_count', by='sentiment', ax=axes[1,0])
    axes[1,0].set_title('Word Count Box Plot by Sentiment')
    axes[1,0].set_xlabel('Sentiment')
    axes[1,0].set_ylabel('Word Count')
    
    # Character count analysis
    axes[1,1].hist(df_in[df_in['sentiment'] == 'positive']['char_count'], bins=50, 
                   alpha=0.6, label='Positive', color='green')
    axes[1,1].hist(df_in[df_in['sentiment'] == 'negative']['char_count'], bins=50, 
                   alpha=0.6, label='Negative', color='red')
    axes[1,1].set_title('Character Count Distribution by Sentiment')
    axes[1,1].set_xlabel('Character Count')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('review_length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("✓ Review length plots saved as 'review_length_analysis.png'")
    
    # Statistical comparison
    from scipy import stats
    pos_words = df_in[df_in['sentiment'] == 'positive']['word_count']
    neg_words = df_in[df_in['sentiment'] == 'negative']['word_count']
    
    t_stat, p_value = stats.ttest_ind(pos_words, neg_words)
    print("\nT-test results:")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

def analyze_word_frequencies(df_in):
    """Analyze and plot word frequencies for positive vs negative reviews"""
    print("\n" + "-"*40)
    print("3. WORD FREQUENCY ANALYSIS")
    print("-"*40)
    
    # Download NLTK data if not already present
    if HAS_NLTK:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
    
    # Get word frequencies by sentiment
    positive_words = []
    negative_words = []
    
    def clean_text_basic(text):
        """Basic text cleaning for frequency analysis"""
        # Remove HTML tags
        if HAS_BS4:
            text = BeautifulSoup(text, 'html.parser').get_text()
        else:
            text = re.sub(r'<[^>]+>', '', text)  # Simple HTML tag removal
        
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        if HAS_NLTK:
            try:
                nltk.data.find('corpora/stopwords')
                stop_words = set(stopwords.words('english'))
            except LookupError:
                print("NLTK stopwords not found, downloading...")
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
        else:
            # Basic stop words list
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return words
    
    for _, row in df_in.iterrows():
        words = clean_text_basic(row['review'])
        if row['sentiment'] == 'positive':
            positive_words.extend(words)
        else:
            negative_words.extend(words)

    # Create word clouds (if available)
    if HAS_WORDCLOUD:
        print("\nGenerating word clouds...")
        # Positive reviews word cloud
        pos_text = ' '.join(positive_words)
        pos_wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Greens').generate(pos_text)
        # Negative reviews word cloud
        neg_text = ' '.join(negative_words)
        neg_wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Reds').generate(neg_text)
        # Plot word clouds
        axes_wc = plt.subplots(1, 2, figsize=(20, 8))[1]
        axes_wc[0].imshow(pos_wordcloud, interpolation='bilinear')
        axes_wc[0].set_title('Positive Reviews Word Cloud')
        axes_wc[0].axis('off')
        axes_wc[1].imshow(neg_wordcloud, interpolation='bilinear')
        axes_wc[1].set_title('Negative Reviews Word Cloud')
        axes_wc[1].axis('off')
        plt.tight_layout()
        plt.savefig('word_clouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Word clouds saved as 'word_clouds.png'")
    else:
        print("WordCloud not available - skipping word cloud generation")
    
    # Count frequencies
    pos_freq = Counter(positive_words).most_common(20)
    neg_freq = Counter(negative_words).most_common(20)
    
    print("Most common words in POSITIVE reviews:")
    for word, count in pos_freq:
        print(f"{word:15}: {count:6d}")
    
    print("\nMost common words in NEGATIVE reviews:")
    for word, count in neg_freq:
        print(f"{word:15}: {count:6d}")
    
    # Plot word frequencies
    axes = plt.subplots(1, 2, figsize=(20, 8))[1]
    
    # Positive words
    pos_words_list, pos_counts = zip(*pos_freq)
    axes[0].barh(range(len(pos_words_list)), pos_counts, color='green', alpha=0.7)
    axes[0].set_yticks(range(len(pos_words_list)))
    axes[0].set_yticklabels(pos_words_list)
    axes[0].set_title('Top 20 Words in Positive Reviews')
    axes[0].set_xlabel('Frequency')
    
    # Negative words
    neg_words_list, neg_counts = zip(*neg_freq)
    axes[1].barh(range(len(neg_words_list)), neg_counts, color='red', alpha=0.7)
    axes[1].set_yticks(range(len(neg_words_list)))
    axes[1].set_yticklabels(neg_words_list)
    axes[1].set_title('Top 20 Words in Negative Reviews')
    axes[1].set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('word_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Word frequency plots saved as 'word_frequency_analysis.png'")
    
    return pos_freq, neg_freq

def dimensionality_reduction_visualization(df_in):
    """Apply PCA, UMAP, and t-SNE to TF-IDF vectors and visualize"""
    print("\n" + "-"*40)
    print("4. DIMENSIONALITY REDUCTION VISUALIZATION")
    print("-"*40)
    
    # Use a sample for faster processing
    sample_size = 5000
    df_sample = df_in.sample(n=sample_size, random_state=42)
    
    print(f"Using sample of {sample_size} reviews for visualization...")
    
    # Clean and vectorize text
    def clean_text_for_tfidf(text):
        if HAS_BS4:
            text = BeautifulSoup(text, 'html.parser').get_text()
        else:
            text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text
    
    df_sample['cleaned_review'] = df_sample['review'].apply(clean_text_for_tfidf)
    
    # TF-IDF Vectorization
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                ngram_range=(1, 2), min_df=5)
    tfidf_matrix = vectorizer.fit_transform(df_sample['cleaned_review'])
    
    # Encode sentiment labels
    y = (df_sample['sentiment'] == 'positive').astype(int)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Apply dimensionality reduction techniques
    print("Applying dimensionality reduction techniques...")
    
    # PCA
    print("  - PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    
    # t-SNE
    print("  - t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(tfidf_matrix.toarray()[:2000])  # Use smaller sample for t-SNE
    
    # UMAP (if available)
    if HAS_UMAP:
        print("  - UMAP...")
        umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        umap_result = umap_model.fit_transform(tfidf_matrix.toarray())
        num_plots = 3
    else:
        print("  - UMAP not available, skipping...")
        umap_result = None
        num_plots = 2
    
    # Visualization
    axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6))[1]
    if num_plots == 2:
        axes = [axes[0], axes[1]]  # Ensure it's a list for consistency
    
    # PCA plot
    scatter1 = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                              c=y, cmap='RdYlGn', alpha=0.6, s=20)
    axes[0].set_title(f'PCA Visualization\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter1, ax=axes[0])
    
    # t-SNE plot
    y_tsne = y[:2000]  # Match the sample size used for t-SNE
    scatter2 = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                              c=y_tsne, cmap='RdYlGn', alpha=0.6, s=20)
    axes[1].set_title('t-SNE Visualization')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    # UMAP plot (if available)
    if HAS_UMAP and umap_result is not None:
        scatter3 = axes[2].scatter(umap_result[:, 0], umap_result[:, 1], 
                                  c=y, cmap='RdYlGn', alpha=0.6, s=20)
        axes[2].set_title('UMAP Visualization')
        axes[2].set_xlabel('UMAP 1')
        axes[2].set_ylabel('UMAP 2')
        plt.colorbar(scatter3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Dimensionality reduction plots saved as 'dimensionality_reduction.png'")
    
    # Print explained variance for PCA
    print("\nPCA Results:")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    return vectorizer, tfidf_matrix

# ===============================================================================
# PHASE 2: FEATURE ENGINEERING & REDUCTION
# ===============================================================================

def create_custom_features(df_in):
    """Create custom features for sentiment analysis"""
    print("\n" + "="*50)
    print("PHASE 2: FEATURE ENGINEERING & REDUCTION")
    print("="*50)
    print("\n" + "-"*40)
    print("1. CUSTOM FEATURE CREATION")
    print("-"*40)
    
    # Create a copy to avoid modifying the original
    df_features = df_in.copy()
    
    # Basic text statistics
    df_features['word_count'] = df_features['review'].apply(lambda x: len(str(x).split()))
    df_features['char_count'] = df_features['review'].apply(len)
    df_features['sentence_count'] = df_features['review'].apply(lambda x: len(re.split(r'[.!?]+', str(x))))
    df_features['avg_word_length'] = df_features['review'].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
    )
    
    # Punctuation and special character counts
    df_features['exclamation_count'] = df_features['review'].apply(lambda x: str(x).count('!'))
    df_features['question_count'] = df_features['review'].apply(lambda x: str(x).count('?'))
    df_features['period_count'] = df_features['review'].apply(lambda x: str(x).count('.'))
    df_features['comma_count'] = df_features['review'].apply(lambda x: str(x).count(','))
    df_features['quote_count'] = df_features['review'].apply(lambda x: str(x).count('"') + str(x).count("'"))
    
    # Capitalization features
    df_features['capital_count'] = df_features['review'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df_features['capital_ratio'] = df_features['capital_count'] / df_features['char_count']
    
    # HTML tags (since these are movie reviews from web)
    df_features['html_tag_count'] = df_features['review'].apply(lambda x: len(re.findall(r'<[^>]+>', str(x))))
    
    # Sentiment lexicon scoring using TextBlob (if available)
    if HAS_TEXTBLOB:
        print("Creating TextBlob sentiment scores...")
        def get_textblob_sentiment(text):
            try:
                blob = TextBlob(str(text))
                return blob.sentiment.polarity, blob.sentiment.subjectivity
            except Exception as ex:  # Intentionally broad, TextBlob can raise various errors
                return 0.0, 0.0
        
        sentiment_scores = df_features['review'].apply(get_textblob_sentiment)
        df_features['textblob_polarity'] = [score[0] for score in sentiment_scores]
        df_features['textblob_subjectivity'] = [score[1] for score in sentiment_scores]
    else:
        print("TextBlob not available, creating basic sentiment scores...")
        # Simple positive/negative word counting
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'perfect', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'boring'}
        
        def simple_sentiment_score(text):
            words = str(text).lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            return (pos_count - neg_count) / len(words) if words else 0
        
        df_features['simple_sentiment_score'] = df_features['review'].apply(simple_sentiment_score)
    
    # Display feature statistics
    feature_cols = [col for col in df_features.columns if col not in ['review', 'sentiment']]
    print(f"\nCreated {len(feature_cols)} custom features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    print("\nFeature statistics by sentiment:")
    stats_summary = df_features.groupby('sentiment')[feature_cols].mean()
    print(stats_summary.round(3))
    
    return df_features, feature_cols

def apply_feature_selection(df_features, feature_cols):
    """Apply Chi-Square and Information Gain for feature selection"""
    print("\n" + "-"*40)
    print("2. FEATURE SELECTION")
    print("-"*40)
    
    # Prepare text for TF-IDF
    def clean_text_simple(text):
        if HAS_BS4:
            text = BeautifulSoup(text, 'html.parser').get_text()
        else:
            text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        return text
    
    df_features['cleaned_text'] = df_features['review'].apply(clean_text_simple)
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2), min_df=5)
    tfidf_features = tfidf.fit_transform(df_features['cleaned_text'])
    
    # Convert to dense array and combine with custom features
    tfidf_dense = tfidf_features.toarray()
    custom_features_array = df_features[feature_cols].values
    
    # Combine all features
    all_features = np.hstack([tfidf_dense, custom_features_array])
    
    # Encode target variable
    y = (df_features['sentiment'] == 'positive').astype(int)
    
    print(f"Total features before selection: {all_features.shape[1]}")
    print(f"TF-IDF features: {tfidf_dense.shape[1]}")
    print(f"Custom features: {custom_features_array.shape[1]}")
    
    # Chi-Square feature selection
    print("\nApplying Chi-Square feature selection...")
    chi2_selector = SelectKBest(chi2, k=500)  # Select top 500 features
    features_chi2 = chi2_selector.fit_transform(all_features, y)
    
    # Information Gain (Mutual Information) feature selection
    print("Applying Information Gain feature selection...")
    mi_selector = SelectKBest(mutual_info_classif, k=500)
    features_mi = mi_selector.fit_transform(all_features, y)
    
    print(f"Features after Chi-Square selection: {features_chi2.shape[1]}")
    print(f"Features after Information Gain selection: {features_mi.shape[1]}")
    
    # Get feature importance scores
    chi2_scores = chi2_selector.scores_
    mi_scores = mi_selector.scores_
    
    # Create feature names
    tfidf_names = [f"tfidf_{i}" for i in range(tfidf_dense.shape[1])]
    all_feature_names = tfidf_names + feature_cols
    
    # Get top features for each method
    chi2_top_indices = np.argsort(chi2_scores)[-20:][::-1]
    mi_top_indices = np.argsort(mi_scores)[-20:][::-1]
    
    print("\nTop 20 features by Chi-Square:")
    for i, idx in enumerate(chi2_top_indices):
        print(f"  {i+1:2d}. {all_feature_names[idx]}: {chi2_scores[idx]:.3f}")
    
    print("\nTop 20 features by Information Gain:")
    for i, idx in enumerate(mi_top_indices):
        print(f"  {i+1:2d}. {all_feature_names[idx]}: {mi_scores[idx]:.3f}")
    
    return {
        'full_features': all_features,
        'chi2_features': features_chi2,
        'mi_features': features_mi,
        'tfidf': tfidf,
        'feature_names': all_feature_names,
        'chi2_selector': chi2_selector,
        'mi_selector': mi_selector,
        'target': y
    }

def compare_feature_sets(feature_data):
    """Compare model performance with different feature sets"""
    print("\n" + "-"*40)
    print("3. FEATURE SET COMPARISON")
    print("-"*40)
    
    X_full = feature_data['full_features']
    X_chi2 = feature_data['chi2_features']
    X_mi = feature_data['mi_features']
    y = feature_data['target']
    
    # Split data
    X_full_train, X_full_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    X_chi2_train, X_chi2_test = train_test_split(
        X_chi2, test_size=0.2, random_state=42, stratify=y
    )[0:2]
    X_mi_train, X_mi_test = train_test_split(
        X_mi, test_size=0.2, random_state=42, stratify=y
    )[0:2]
    
    # Test different models with different feature sets
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    feature_sets = {
        'Full Features': (X_full_train, X_full_test),
        'Chi-Square Selected': (X_chi2_train, X_chi2_test),
        'Information Gain Selected': (X_mi_train, X_mi_test)
    }
    
    results = []
    
    print("\nComparing feature sets with different models:")
    print("-" * 70)
    print(f"{'Model':<20} {'Feature Set':<25} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for model_name, model in models.items():
        for feature_name, (X_train_fs, X_test_fs) in feature_sets.items():
            # Train model
            model.fit(X_train_fs, y_train)
            
            # Predict
            y_pred = model.predict(X_test_fs)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                'model': model_name,
                'feature_set': feature_name,
                'accuracy': accuracy,
                'f1_score': f1
            })
            
            print(f"{model_name:<20} {feature_name:<25} {accuracy:<10.4f} {f1:<10.4f}")
    
    print("-" * 70)
    
    # Create visualization of results
    results_df = pd.DataFrame(results)
    
    axes = plt.subplots(1, 2, figsize=(15, 6))[1]
    
    # Accuracy comparison
    pivot_acc = results_df.pivot(index='model', columns='feature_set', values='accuracy')
    pivot_acc.plot(kind='bar', ax=axes[0], rot=45)
    axes[0].set_title('Accuracy Comparison by Feature Set')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(title='Feature Set', loc='lower right')
    
    # F1-Score comparison
    pivot_f1 = results_df.pivot(index='model', columns='feature_set', values='f1_score')
    pivot_f1.plot(kind='bar', ax=axes[1], rot=45)
    axes[1].set_title('F1-Score Comparison by Feature Set')
    axes[1].set_ylabel('F1-Score')
    axes[1].legend(title='Feature Set', loc='lower right')
    
    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature comparison plots saved as 'feature_comparison.png'")
    
    return results_df, feature_data

# ===============================================================================
# MAIN EXECUTION - PHASE 1 (EDA)
# ===============================================================================

def run_phase1_eda():
    """Run Phase 1: Exploratory Data Analysis"""
    try:
        # Execute Phase 1: EDA
    df_local = load_and_explore_data()
    df_local = calculate_review_lengths(df_local)
    plot_review_length_histograms(df_local)
    pos_freq, neg_freq = analyze_word_frequencies(df_local)
    vectorizer, tfidf_matrix = dimensionality_reduction_visualization(df_local)
        
        print("\n" + "="*80)
        print("PHASE 1 (EDA) COMPLETED!")
        print("="*80)
        
        return df, pos_freq, neg_freq, vectorizer, tfidf_matrix
    
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        return None

def run_phase2_feature_engineering():
    """Run Phase 2: Feature Engineering & Reduction"""
    try:
        # Load data for Phase 2
        df_local = pd.read_csv('IMDB Dataset.csv')
        # Execute Phase 2: Feature Engineering
        df_features, feature_cols = create_custom_features(df_local)
        feature_data = apply_feature_selection(df_features, feature_cols)
        results_df, feature_data = compare_feature_sets(feature_data)
        print("\n" + "="*80)
        print("PHASE 2 (FEATURE ENGINEERING) COMPLETED!")
        print("="*80)
        return df_features, feature_cols, feature_data, results_df
    except Exception as e:  # Intentionally broad for phase error reporting
        print(f"Error in Phase 2: {e}")
        return None

if __name__ == "__main__":
    print("Testing basic functionality...")
    try:
        print("✓ Basic libraries imported successfully")
        df_main = pd.read_csv('IMDB Dataset.csv')
        print(f"✓ Dataset loaded successfully: {df_main.shape}")
        print("\n" + "="*50)
        print("STARTING PHASE 1: EDA")
        print("="*50)
        phase1_results = run_phase1_eda()
        if phase1_results is not None:
            print("\n" + "="*50)
            print("STARTING PHASE 2: FEATURE ENGINEERING")
            print("="*50)
            phase2_results = run_phase2_feature_engineering()
            if phase2_results is not None:
                print("\n" + "="*80)
                print("PHASES 1 & 2 COMPLETED SUCCESSFULLY!")
                print("Generated files:")
                print("- review_length_analysis.png")
                print("- word_frequency_analysis.png") 
                print("- dimensionality_reduction.png")
                print("- feature_comparison.png")
                print("="*80)
                print("Next: Run Phase 3 - Text Preprocessing & Model Training")
    except Exception as e:  # Intentionally broad for main error reporting
        print(f"Error: {e}")
        print("Please check that all required packages are installed.")
