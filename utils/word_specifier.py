# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Word Specifier utilities for text processing and classification."""

import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml


class WordSpecifier:
    """
    A comprehensive word specifier for text processing, classification, and filtering.
    
    Supports various word categorization methods including:
    - Part-of-speech tagging
    - Word frequency analysis
    - Custom word lists and categories
    - Text preprocessing and normalization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the WordSpecifier with optional configuration.
        
        Args:
            config_path: Path to YAML configuration file with word categories
        """
        self.word_categories = {}
        self.stop_words = set()
        self.custom_words = {}
        self.frequency_dict = {}
        
        # Load default configuration
        self._load_default_config()
        
        # Load custom configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self):
        """Load default word categories and stop words."""
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
        # Default word categories
        self.word_categories = {
            'positive': {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'awesome', 'brilliant', 'perfect', 'outstanding', 'superb', 'magnificent'
            },
            'negative': {
                'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'pathetic',
                'disappointing', 'poor', 'worst', 'hate', 'dislike', 'annoying'
            },
            'technical': {
                'algorithm', 'model', 'neural', 'network', 'training', 'validation',
                'accuracy', 'precision', 'recall', 'loss', 'optimization', 'gradient'
            },
            'colors': {
                'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
                'black', 'white', 'gray', 'brown', 'cyan', 'magenta'
            },
            'numbers': {
                'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                'eight', 'nine', 'ten', 'hundred', 'thousand', 'million'
            }
        }
    
    def load_config(self, config_path: str):
        """
        Load word categories from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'word_categories' in config:
                for category, words in config['word_categories'].items():
                    self.word_categories[category] = set(words)
            
            if 'stop_words' in config:
                self.stop_words.update(config['stop_words'])
            
            if 'custom_words' in config:
                self.custom_words.update(config['custom_words'])
                
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    def preprocess_text(self, text: str, 
                       lowercase: bool = True,
                       remove_punctuation: bool = True,
                       remove_numbers: bool = False,
                       remove_stop_words: bool = False) -> str:
        """
        Preprocess text with various normalization options.
        
        Args:
            text: Input text to preprocess
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            remove_stop_words: Remove common stop words
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stop words
        if remove_stop_words:
            words = text.split()
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens/words
        """
        # Simple word tokenization
        text = self.preprocess_text(text, remove_punctuation=True)
        return text.split()
    
    def categorize_words(self, words: Union[str, List[str]]) -> Dict[str, List[str]]:
        """
        Categorize words based on predefined categories.
        
        Args:
            words: Single word string or list of words
            
        Returns:
            Dictionary mapping categories to lists of matching words
        """
        if isinstance(words, str):
            words = self.tokenize(words)
        
        categorized = {category: [] for category in self.word_categories.keys()}
        categorized['uncategorized'] = []
        
        for word in words:
            word_lower = word.lower()
            found_category = False
            
            for category, category_words in self.word_categories.items():
                if word_lower in category_words:
                    categorized[category].append(word)
                    found_category = True
                    break
            
            if not found_category:
                categorized['uncategorized'].append(word)
        
        return categorized
    
    def get_word_frequency(self, text: str, top_n: Optional[int] = None) -> Dict[str, int]:
        """
        Calculate word frequency in text.
        
        Args:
            text: Input text to analyze
            top_n: Return only top N most frequent words
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        words = self.tokenize(text)
        frequency = {}
        
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.stop_words:
                frequency[word_lower] = frequency.get(word_lower, 0) + 1
        
        # Sort by frequency
        sorted_freq = dict(sorted(frequency.items(), key=lambda x: x[1], reverse=True))
        
        if top_n:
            return dict(list(sorted_freq.items())[:top_n])
        
        return sorted_freq
    
    def filter_words(self, words: Union[str, List[str]], 
                    categories: Optional[List[str]] = None,
                    min_length: int = 1,
                    max_length: Optional[int] = None,
                    exclude_stop_words: bool = True) -> List[str]:
        """
        Filter words based on various criteria.
        
        Args:
            words: Input words to filter
            categories: Only include words from these categories
            min_length: Minimum word length
            max_length: Maximum word length
            exclude_stop_words: Exclude common stop words
            
        Returns:
            List of filtered words
        """
        if isinstance(words, str):
            words = self.tokenize(words)
        
        filtered = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check length constraints
            if len(word) < min_length:
                continue
            if max_length and len(word) > max_length:
                continue
            
            # Check stop words
            if exclude_stop_words and word_lower in self.stop_words:
                continue
            
            # Check categories
            if categories:
                word_in_category = False
                for category in categories:
                    if category in self.word_categories and word_lower in self.word_categories[category]:
                        word_in_category = True
                        break
                if not word_in_category:
                    continue
            
            filtered.append(word)
        
        return filtered
    
    def find_similar_words(self, target_word: str, words: List[str], 
                          similarity_threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find words similar to target word using simple string similarity.
        
        Args:
            target_word: Word to find similarities for
            words: List of words to search in
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of tuples (word, similarity_score)
        """
        def simple_similarity(word1: str, word2: str) -> float:
            """Calculate simple character-based similarity."""
            if not word1 or not word2:
                return 0.0
            
            # Longest common subsequence approach
            m, n = len(word1), len(word2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n] / max(m, n)
        
        similar_words = []
        target_lower = target_word.lower()
        
        for word in words:
            if word.lower() == target_lower:
                continue
            
            similarity = simple_similarity(target_lower, word.lower())
            if similarity >= similarity_threshold:
                similar_words.append((word, similarity))
        
        return sorted(similar_words, key=lambda x: x[1], reverse=True)
    
    def add_category(self, category_name: str, words: List[str]):
        """
        Add a new word category.
        
        Args:
            category_name: Name of the new category
            words: List of words in the category
        """
        self.word_categories[category_name] = set(words)
    
    def add_words_to_category(self, category_name: str, words: List[str]):
        """
        Add words to an existing category.
        
        Args:
            category_name: Name of the category
            words: List of words to add
        """
        if category_name not in self.word_categories:
            self.word_categories[category_name] = set()
        
        self.word_categories[category_name].update(words)
    
    def remove_words_from_category(self, category_name: str, words: List[str]):
        """
        Remove words from a category.
        
        Args:
            category_name: Name of the category
            words: List of words to remove
        """
        if category_name in self.word_categories:
            self.word_categories[category_name] -= set(words)
    
    def get_category_words(self, category_name: str) -> Set[str]:
        """
        Get all words in a specific category.
        
        Args:
            category_name: Name of the category
            
        Returns:
            Set of words in the category
        """
        return self.word_categories.get(category_name, set())
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories."""
        return list(self.word_categories.keys())
    
    def export_config(self, output_path: str):
        """
        Export current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file
        """
        config = {
            'word_categories': {k: list(v) for k, v in self.word_categories.items()},
            'stop_words': list(self.stop_words),
            'custom_words': self.custom_words
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        words = self.tokenize(text)
        categorized = self.categorize_words(words)
        frequency = self.get_word_frequency(text, top_n=10)
        
        analysis = {
            'total_words': len(words),
            'unique_words': len(set(word.lower() for word in words)),
            'categorized_words': categorized,
            'top_words': frequency,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'longest_word': max(words, key=len) if words else "",
            'shortest_word': min(words, key=len) if words else ""
        }
        
        return analysis


def create_default_word_config(output_path: str = "data/word_config.yaml"):
    """
    Create a default word configuration file.
    
    Args:
        output_path: Path to save the configuration file
    """
    specifier = WordSpecifier()
    specifier.export_config(output_path)
    print(f"Default word configuration saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    specifier = WordSpecifier()
    
    # Example text analysis
    sample_text = """
    This is a great example of text processing with the YOLOv5 word specifier.
    The algorithm can categorize words, analyze frequency, and filter content.
    Machine learning models like neural networks require good data preprocessing.
    """
    
    print("=== Word Specifier Example ===")
    analysis = specifier.analyze_text(sample_text)
    
    print(f"Total words: {analysis['total_words']}")
    print(f"Unique words: {analysis['unique_words']}")
    print(f"Average word length: {analysis['avg_word_length']:.2f}")
    
    print("\nCategorized words:")
    for category, words in analysis['categorized_words'].items():
        if words:
            print(f"  {category}: {words}")
    
    print("\nTop words:")
    for word, freq in analysis['top_words'].items():
        print(f"  {word}: {freq}")
    
    # Create default config
    create_default_word_config()