# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Word utilities for YOLOv5 text processing integration."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .word_specifier import WordSpecifier


def load_class_names_with_words(names_file: str, word_config: Optional[str] = None) -> Dict[int, Dict[str, str]]:
    """
    Load class names and associate them with word categories.
    
    Args:
        names_file: Path to class names file (YAML or JSON)
        word_config: Path to word configuration file
        
    Returns:
        Dictionary mapping class indices to class info with word categories
    """
    import yaml
    
    # Load class names
    if names_file.endswith('.yaml') or names_file.endswith('.yml'):
        with open(names_file, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
            else:
                names = data
    elif names_file.endswith('.json'):
        with open(names_file, 'r') as f:
            names = json.load(f)
    else:
        # Assume it's a simple text file with one name per line
        with open(names_file, 'r') as f:
            names = [line.strip() for line in f.readlines()]
    
    # Initialize word specifier
    word_specifier = WordSpecifier(word_config)
    
    # Process class names
    class_info = {}
    for i, name in enumerate(names):
        if isinstance(name, dict):
            class_name = name.get('name', str(i))
        else:
            class_name = str(name)
        
        # Categorize the class name
        categorized = word_specifier.categorize_words(class_name)
        primary_category = 'uncategorized'
        for category, words in categorized.items():
            if words and category != 'uncategorized':
                primary_category = category
                break
        
        class_info[i] = {
            'name': class_name,
            'category': primary_category,
            'processed_name': word_specifier.preprocess_text(class_name),
            'word_count': len(word_specifier.tokenize(class_name))
        }
    
    return class_info


def create_word_based_class_mapping(class_names: List[str], 
                                   word_config: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Create a mapping from word categories to class indices.
    
    Args:
        class_names: List of class names
        word_config: Path to word configuration file
        
    Returns:
        Dictionary mapping categories to lists of class indices
    """
    word_specifier = WordSpecifier(word_config)
    category_mapping = {}
    
    for i, class_name in enumerate(class_names):
        categorized = word_specifier.categorize_words(class_name)
        
        for category, words in categorized.items():
            if words:  # If class name has words in this category
                if category not in category_mapping:
                    category_mapping[category] = []
                category_mapping[category].append(i)
    
    return category_mapping


def filter_predictions_by_words(predictions: List[Dict], 
                               word_categories: List[str],
                               class_names: List[str],
                               word_config: Optional[str] = None) -> List[Dict]:
    """
    Filter predictions based on word categories of class names.
    
    Args:
        predictions: List of prediction dictionaries
        word_categories: Categories to keep
        class_names: List of class names
        word_config: Path to word configuration file
        
    Returns:
        Filtered predictions
    """
    word_specifier = WordSpecifier(word_config)
    
    # Create category mapping
    category_mapping = create_word_based_class_mapping(class_names, word_config)
    
    # Get class indices for desired categories
    keep_classes = set()
    for category in word_categories:
        if category in category_mapping:
            keep_classes.update(category_mapping[category])
    
    # Filter predictions
    filtered_predictions = []
    for pred in predictions:
        if pred.get('class', -1) in keep_classes:
            filtered_predictions.append(pred)
    
    return filtered_predictions


def enhance_class_names_with_synonyms(class_names: List[str],
                                    word_config: Optional[str] = None) -> Dict[int, List[str]]:
    """
    Enhance class names with synonyms and related words.
    
    Args:
        class_names: Original class names
        word_config: Path to word configuration file
        
    Returns:
        Dictionary mapping class indices to lists of synonyms
    """
    word_specifier = WordSpecifier(word_config)
    enhanced_names = {}
    
    # Get all words from all categories
    all_words = []
    for category_words in word_specifier.word_categories.values():
        all_words.extend(category_words)
    
    for i, class_name in enumerate(class_names):
        synonyms = [class_name]  # Start with original name
        
        # Find similar words
        words = word_specifier.tokenize(class_name)
        for word in words:
            similar = word_specifier.find_similar_words(word, all_words, similarity_threshold=0.7)
            synonyms.extend([sim_word for sim_word, _ in similar[:3]])  # Add top 3 similar words
        
        enhanced_names[i] = list(set(synonyms))  # Remove duplicates
    
    return enhanced_names


def generate_text_augmentations(text: str, 
                               num_augmentations: int = 5,
                               word_config: Optional[str] = None) -> List[str]:
    """
    Generate text augmentations by replacing words with synonyms.
    
    Args:
        text: Original text
        num_augmentations: Number of augmentations to generate
        word_config: Path to word configuration file
        
    Returns:
        List of augmented texts
    """
    word_specifier = WordSpecifier(word_config)
    augmentations = [text]  # Include original
    
    words = word_specifier.tokenize(text)
    all_category_words = []
    for category_words in word_specifier.word_categories.values():
        all_category_words.extend(category_words)
    
    for _ in range(num_augmentations):
        augmented_words = words.copy()
        
        # Replace some words with similar ones
        for i, word in enumerate(words):
            if word.lower() not in word_specifier.stop_words:
                similar = word_specifier.find_similar_words(
                    word, all_category_words, similarity_threshold=0.6
                )
                if similar:
                    # Randomly pick a similar word
                    import random
                    augmented_words[i] = random.choice(similar)[0]
        
        augmented_text = ' '.join(augmented_words)
        if augmented_text not in augmentations:
            augmentations.append(augmented_text)
    
    return augmentations[:num_augmentations + 1]


def create_word_embedding_mapping(class_names: List[str],
                                 embedding_dim: int = 300,
                                 word_config: Optional[str] = None) -> Dict[int, List[float]]:
    """
    Create simple word embeddings for class names (placeholder implementation).
    
    Args:
        class_names: List of class names
        embedding_dim: Dimension of embeddings
        word_config: Path to word configuration file
        
    Returns:
        Dictionary mapping class indices to embedding vectors
    """
    import numpy as np
    
    word_specifier = WordSpecifier(word_config)
    embeddings = {}
    
    for i, class_name in enumerate(class_names):
        # Simple embedding based on word characteristics
        words = word_specifier.tokenize(class_name)
        
        # Create features based on word properties
        features = []
        
        # Length features
        features.append(len(class_name))
        features.append(len(words))
        features.append(np.mean([len(w) for w in words]) if words else 0)
        
        # Category features
        categorized = word_specifier.categorize_words(words)
        for category in word_specifier.get_all_categories():
            features.append(len(categorized.get(category, [])))
        
        # Pad or truncate to desired dimension
        if len(features) < embedding_dim:
            features.extend([0.0] * (embedding_dim - len(features)))
        else:
            features = features[:embedding_dim]
        
        embeddings[i] = features
    
    return embeddings


def export_word_analysis_report(texts: List[str], 
                               output_file: str,
                               word_config: Optional[str] = None):
    """
    Export a comprehensive word analysis report.
    
    Args:
        texts: List of texts to analyze
        output_file: Path to save the report
        word_config: Path to word configuration file
    """
    from .text_processor import TextProcessor
    
    processor = TextProcessor(word_config)
    analysis = processor.analyze_dataset_text(texts)
    
    # Create detailed report
    report = {
        'summary': {
            'total_texts': len(texts),
            'total_words': analysis['total_words'],
            'unique_words': analysis['unique_words'],
            'average_text_length': analysis['avg_text_length'],
            'average_word_length': analysis['avg_word_length']
        },
        'word_categories': analysis['categorized_words'],
        'top_words': analysis['top_words'],
        'category_distribution': analysis.get('category_distribution', {}),
        'quality_metrics': {
            'empty_texts': analysis.get('empty_texts', 0),
            'longest_word': analysis['longest_word'],
            'shortest_word': analysis['shortest_word']
        }
    }
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Word analysis report saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    sample_class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee"
    ]
    
    print("=== Word Utils Example ===")
    
    # Create category mapping
    category_mapping = create_word_based_class_mapping(sample_class_names)
    print("Category mapping:")
    for category, indices in category_mapping.items():
        class_names_in_category = [sample_class_names[i] for i in indices]
        print(f"  {category}: {class_names_in_category}")
    
    # Generate text augmentations
    sample_text = "a person riding a red bicycle"
    augmentations = generate_text_augmentations(sample_text, num_augmentations=3)
    print(f"\nText augmentations for '{sample_text}':")
    for i, aug in enumerate(augmentations):
        print(f"  {i}: {aug}")