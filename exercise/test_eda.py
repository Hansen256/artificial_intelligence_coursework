"""
Simple test of the sentiment analysis - Phase 1 EDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

print("="*60)
print("SENTIMENT ANALYSIS - BASIC EDA TEST")
print("="*60)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Basic stats
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Calculate word counts
print("\nCalculating word counts...")
df['word_count'] = df['review'].apply(lambda x: len(str(x).split()))

print(f"Average word count: {df['word_count'].mean():.2f}")
print(f"By sentiment:")
sentiment_stats = df.groupby('sentiment')['word_count'].mean()
print(sentiment_stats)

# Simple histogram
print("\nCreating basic histogram...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['word_count'], bins=50, alpha=0.7)
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
pos_words = df[df['sentiment'] == 'positive']['word_count']
neg_words = df[df['sentiment'] == 'negative']['word_count']
plt.hist(pos_words, bins=50, alpha=0.6, label='Positive', color='green')
plt.hist(neg_words, bins=50, alpha=0.6, label='Negative', color='red')
plt.title('Word Count by Sentiment')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

print("Basic EDA completed successfully!")