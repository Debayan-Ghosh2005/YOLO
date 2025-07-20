# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Text processing utilities using WordSpecifier for YOLOv5."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .word_specifier import WordSpecifier


class TextProcessor:
    """
    Advanced text processor for YOLOv5 that uses WordSpecifier for text analysis and processing.
    
    Useful for processing image captions, labels, metadata, and other text data
    associated with computer vision tasks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TextProcessor with WordSpecifier.
        
        Args:
            config_path: Path to word configuration file
        """
        self.word_specifier = WordSpecifier(config_path)
        self.processed_texts = []
        self.text_stats = {}
    
    def process_image_labels(self, labels: List[str], 
                           normalize: bool = True,
                           filter_categories: Optional[List[str]] = None) -> List[str]:
        """
        Process image labels/captions for better consistency.
        
        Args:
            labels: List of text labels
            normalize: Apply text normalization
            filter_categories: Only keep words from these categories
            
        Returns:
            List of processed labels
        """
        processed = []
        
        for label in labels:
            if normalize:
                # Normalize text
                processed_text = self.word_specifier.preprocess_text(
                    label, 
                    lowercase=True,
                    remove_punctuation=True,
                    remove_stop_words=True
                )
            else:
                processed_text = label
            
            # Filter by categories if specified
            if filter_categories:
                words = self.word_specifier.tokenize(processed_text)
                filtered_words = self.word_specifier.filter_words(
                    words, 
                    categories=filter_categories
                )
                processed_text = ' '.join(filtered_words)
            
            processed.append(processed_text)
        
        return processed
    
    def extract_keywords(self, text: str, 
                        top_n: int = 10,
                        categories: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """
        Extract keywords from text based on frequency and categories.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            categories: Only consider words from these categories
            
        Returns:
            List of (keyword, frequency) tuples
        """
        # Get word frequency
        frequency = self.word_specifier.get_word_frequency(text, top_n=None)
        
        # Filter by categories if specified
        if categories:
            categorized = self.word_specifier.categorize_words(list(frequency.keys()))
            category_words = set()
            for cat in categories:
                if cat in categorized:
                    category_words.update(categorized[cat])
            
            frequency = {word: freq for word, freq in frequency.items() 
                        if word in category_words}
        
        # Return top N
        sorted_keywords = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
    
    def analyze_dataset_text(self, text_data: List[str]) -> Dict:
        """
        Analyze text data from a dataset (e.g., image captions, labels).
        
        Args:
            text_data: List of text strings to analyze
            
        Returns:
            Dictionary with comprehensive analysis
        """
        all_text = ' '.join(text_data)
        analysis = self.word_specifier.analyze_text(all_text)
        
        # Additional dataset-specific analysis
        analysis['num_samples'] = len(text_data)
        analysis['avg_text_length'] = sum(len(text) for text in text_data) / len(text_data)
        analysis['empty_texts'] = sum(1 for text in text_data if not text.strip())
        
        # Category distribution
        category_counts = {}
        for text in text_data:
            categorized = self.word_specifier.categorize_words(text)
            for category, words in categorized.items():
                if words:  # Only count non-empty categories
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        analysis['category_distribution'] = category_counts
        
        return analysis
    
    def clean_annotation_text(self, text: str) -> str:
        """
        Clean annotation text for better processing.
        
        Args:
            text: Raw annotation text
            
        Returns:
            Cleaned text
        """
        # Remove common annotation artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        text = re.sub(r'<.*?>', '', text)    # Remove HTML-like tags
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = re.sub(r'#\w+', '', text)     # Remove hashtags
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return self.word_specifier.preprocess_text(text)
    
    def generate_text_summary(self, texts: List[str], max_words: int = 50) -> str:
        """
        Generate a summary from multiple text inputs.
        
        Args:
            texts: List of text strings
            max_words: Maximum words in summary
            
        Returns:
            Generated summary text
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Get most frequent meaningful words
        keywords = self.extract_keywords(combined_text, top_n=max_words)
        
        # Create summary from keywords
        summary_words = [word for word, _ in keywords]
        return ' '.join(summary_words)
    
    def validate_text_quality(self, text: str) -> Dict[str, bool]:
        """
        Validate text quality for dataset consistency.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'has_content': bool(text.strip()),
            'reasonable_length': 5 <= len(text.split()) <= 100,
            'no_excessive_punctuation': text.count('!') + text.count('?') <= 3,
            'no_all_caps': not text.isupper(),
            'has_meaningful_words': False,
            'no_excessive_repetition': True
        }
        
        # Check for meaningful words (not all stop words)
        words = self.word_specifier.tokenize(text)
        meaningful_words = [w for w in words if w.lower() not in self.word_specifier.stop_words]
        validation['has_meaningful_words'] = len(meaningful_words) > 0
        
        # Check for excessive repetition
        if words:
            word_counts = {}
            for word in words:
                word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
            max_repetition = max(word_counts.values())
            validation['no_excessive_repetition'] = max_repetition <= len(words) * 0.3
        
        return validation
    
    def suggest_text_improvements(self, text: str) -> List[str]:
        """
        Suggest improvements for text quality.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        validation = self.validate_text_quality(text)
        
        if not validation['has_content']:
            suggestions.append("Add meaningful content to the text")
        
        if not validation['reasonable_length']:
            word_count = len(text.split())
            if word_count < 5:
                suggestions.append("Text is too short, consider adding more descriptive words")
            else:
                suggestions.append("Text is too long, consider summarizing")
        
        if not validation['no_excessive_punctuation']:
            suggestions.append("Reduce excessive punctuation marks")
        
        if not validation['no_all_caps']:
            suggestions.append("Avoid using all capital letters")
        
        if not validation['has_meaningful_words']:
            suggestions.append("Add more meaningful words beyond common stop words")
        
        if not validation['no_excessive_repetition']:
            suggestions.append("Reduce word repetition for better readability")
        
        return suggestions
    
    def batch_process_texts(self, texts: List[str], 
                           operations: List[str] = None) -> List[str]:
        """
        Process multiple texts with specified operations.
        
        Args:
            texts: List of texts to process
            operations: List of operations to apply
            
        Returns:
            List of processed texts
        """
        if operations is None:
            operations = ['normalize', 'clean']
        
        processed = []
        
        for text in texts:
            current_text = text
            
            for operation in operations:
                if operation == 'normalize':
                    current_text = self.word_specifier.preprocess_text(current_text)
                elif operation == 'clean':
                    current_text = self.clean_annotation_text(current_text)
                elif operation == 'remove_stop_words':
                    words = self.word_specifier.tokenize(current_text)
                    filtered = [w for w in words if w.lower() not in self.word_specifier.stop_words]
                    current_text = ' '.join(filtered)
            
            processed.append(current_text)
        
        return processed


def process_dataset_annotations(annotation_file: str, output_file: str = None):
    """
    Process annotations from a dataset file.
    
    Args:
        annotation_file: Path to annotation file
        output_file: Path to save processed annotations
    """
    processor = TextProcessor()
    
    # This is a placeholder - actual implementation would depend on annotation format
    print(f"Processing annotations from {annotation_file}")
    
    # Example processing workflow
    sample_texts = [
        "A person riding a bicycle on the street",
        "Multiple cars parked in a parking lot",
        "A dog running in the park with children playing",
        "Traffic lights showing red signal at intersection"
    ]
    
    # Analyze the texts
    analysis = processor.analyze_dataset_text(sample_texts)
    print("Dataset Analysis:")
    print(f"  Total samples: {analysis['num_samples']}")
    print(f"  Average text length: {analysis['avg_text_length']:.2f}")
    print(f"  Unique words: {analysis['unique_words']}")
    
    # Process the texts
    processed = processor.batch_process_texts(sample_texts)
    print("\nProcessed texts:")
    for original, processed_text in zip(sample_texts, processed):
        print(f"  Original: {original}")
        print(f"  Processed: {processed_text}")
        print()


if __name__ == "__main__":
    # Example usage
    process_dataset_annotations("sample_annotations.txt")