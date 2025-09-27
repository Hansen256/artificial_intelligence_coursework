# Sentiment Analysis Assignment - IMDb Movie Reviews

This project implements a comprehensive sentiment analysis system for movie reviews using machine learning techniques.

## Project Overview

This assignment addresses the following requirements:
- **Dataset**: IMDb Movie Reviews
- **Tasks**: EDA, Feature Engineering, Model Training, Evaluation, and Reflection

## Features Implemented

### Phase 1: Exploratory Data Analysis (EDA)
- Average review length analysis (words)
- Histogram of review lengths by sentiment
- Word frequency plots for positive vs negative reviews
- Scatter plots of PCA-reduced TF-IDF vectors
- Comparison with t-SNE dimensionality reduction

### Phase 2: Feature Engineering & Reduction
- Custom features creation:
  - Text statistics (character count, sentence count, average word length)
  - Punctuation analysis (exclamation marks, question marks)
  - Capitalization patterns
  - Sentiment lexicon scores using TextBlob
  - HTML tag counts
- Chi-Square feature selection for vocabulary reduction
- Comparison of full vs reduced feature sets

### Phase 3: Text Preprocessing
- Text cleaning (HTML removal, special characters)
- Tokenization and normalization
- TF-IDF vectorization with n-grams

### Phase 4: Model Training & Ensemble
- Individual models:
  - Naïve Bayes (MultinomialNB)
  - Logistic Regression
  - Support Vector Machine (SVM)
- Ensemble method: Soft Voting Classifier

### Phase 5: Evaluation & Analysis
- Comprehensive metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrices visualization
- ROC curves comparison
- Error analysis of misclassified reviews

### Phase 6: Reflection
- Discussion of bag-of-words limitations
- Benefits of ensemble methods
- Impact of feature engineering

## Requirements

### Python Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob
```

### Data
- Place your IMDb dataset as `IMDB Dataset.csv` in the project directory
- Expected format: CSV with columns `review` and `sentiment`

## Usage

1. Ensure the dataset is in the correct location:
   ```
   IMDB Dataset.csv
   ```

2. Run the analysis:
   ```bash
   python assignment2.py
   ```

## Output Files

The program generates the following visualization files:
- `review_length_analysis.png` - Review length distributions
- `word_frequency_analysis.png` - Word frequency comparisons
- `dimensionality_reduction_analysis.png` - PCA and t-SNE visualizations
- `feature_importance.png` - Feature importance rankings
- `confusion_matrices.png` - Model confusion matrices
- `roc_curves.png` - ROC curve comparisons

## Code Quality Features

- **Error Handling**: Robust handling of missing data and edge cases
- **Scalability**: Automatic adjustment for small datasets
- **Memory Efficiency**: Sampling for large datasets in visualization
- **Non-interactive Backend**: Works in server environments
- **Comprehensive Logging**: Detailed progress reporting

## Technical Highlights

- **Dynamic t-SNE**: Automatically adjusts perplexity for small datasets
- **Feature Scaling**: MinMax scaling for MultinomialNB compatibility  
- **Flexible TF-IDF**: Vocabulary size adapts to dataset
- **Zero Division Handling**: Prevents division by zero in metrics
- **Exception Management**: Graceful degradation on errors

## Assignment Requirements Fulfilled

✅ **EDA**: Review length analysis, word frequency, dimensionality reduction  
✅ **Feature Engineering**: 10+ custom features, Chi-Square selection  
✅ **Preprocessing**: Text cleaning, tokenization, TF-IDF  
✅ **Models**: Naïve Bayes, Logistic Regression, SVM, Ensemble  
✅ **Evaluation**: All requested metrics plus visualizations  
✅ **Analysis**: Error analysis and model interpretation  
✅ **Reflection**: Limitations and improvements discussion  

## Author

This implementation provides a complete solution for the sentiment analysis coursework assignment with production-ready code quality and comprehensive error handling.