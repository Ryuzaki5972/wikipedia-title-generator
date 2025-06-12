import pandas as pd
import numpy as np
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.data import find


# Download required NLTK resources if not already available
def safe_nltk_download():
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

safe_nltk_download()

class DataPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_dataset(self, train_path, test_path):
        """Load dataset from given paths"""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Extract validation set from training set
        val_df = train_df.sample(n=500, random_state=42)
        train_df = train_df.drop(val_df.index)
        
        return train_df, val_df, test_df
    
    def preprocess_text(self, text):
        """Apply preprocessing steps to text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove non-ASCII characters and punctuation
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # Apply stemming
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df):
        """Apply preprocessing to both title and text columns"""
        processed_df = df.copy()
        processed_df['text'] = processed_df['text'].apply(self.preprocess_text)
        processed_df['title'] = processed_df['title'].apply(self.preprocess_text)
        return processed_df
    
    def process_all_data(self, train_df, val_df, test_df):
        """Process all datasets"""
        train_processed = self.preprocess_dataframe(train_df)
        val_processed = self.preprocess_dataframe(val_df)
        test_processed = self.preprocess_dataframe(test_df)
        return train_processed, val_processed, test_processed

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load datasets
    train_df, val_df, test_df = preprocessor.load_dataset('data/train.csv', 'data/test.csv')
    
    # Process datasets
    train_processed, val_processed, test_processed = preprocessor.process_all_data(train_df, val_df, test_df)
    
    # Save processed datasets
    train_processed.to_csv('train_processed.csv', index=False)
    val_processed.to_csv('val_processed.csv', index=False)
    test_processed.to_csv('test_processed.csv', index=False)
    
    print(f"Processed datasets saved. Train: {len(train_processed)}, Val: {len(val_processed)}, Test: {len(test_processed)}")
    
    # Also save raw datasets for transformer tasks
    train_df.to_csv('train_raw.csv', index=False)
    val_df.to_csv('val_raw.csv', index=False)
    test_df.to_csv('test_raw.csv', index=False)

if __name__ == "__main__":
    main()
