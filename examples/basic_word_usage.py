#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Basic Word Specifier Usage Examples

This script shows simple, practical examples of using the word specifier system.
Run this to get started quickly with the word processing capabilities.

Usage:
    python examples/basic_word_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.word_specifier import WordSpecifier
from utils.text_processor import TextProcessor
from utils.word_utils import create_word_based_class_mapping, generate_text_augmentations


def example_1_basic_analysis():
    """Example 1: Basic text analysis"""
    print("=" * 50)
    print("EXAMPLE 1: Basic Text Analysis")
    print("=" * 50)
    
    specifier = WordSpecifier()
    
    # Sample text about computer vision
    text = """
    YOLOv5 is an excellent object detection model that can identify 
    cars, people, animals, and various objects with amazing accuracy.
    The neural network processes images quickly and provides reliable results.
    """
    
    # Analyze the text
    analysis = specifier.analyze_text(text)
    
    print("Text Analysis Results:")
    print(f"  📊 Total words: {analysis['total_words']}")
    print(f"  🔤 Unique words: {analysis['unique_words']}")
    print(f"  📏 Average word length: {analysis['avg_word_length']:.1f}")
    print(f"  📈 Longest word: '{analysis['longest_word']}'")
    
    print("\n🏷️  Word Categories Found:")
    for category, words in analysis['categorized_words'].items():
        if words:
            print(f"  {category.title()}: {words}")
    
    print("\n🔥 Most Frequent Words:")
    for word, count in list(analysis['top_words'].items())[:5]:
        print(f"  '{word}': {count} times")


def example_2_image_labels():
    """Example 2: Processing image labels"""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Image Label Processing")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # Sample image labels/captions
    image_labels = [
        "A person riding a red bicycle on the street",
        "Multiple cars parked in the parking lot",
        "A dog running happily in the green park",
        "Traffic lights showing red signal",
        "Children playing soccer on the grass"
    ]
    
    print("Original Labels:")
    for i, label in enumerate(image_labels, 1):
        print(f"  {i}. {label}")
    
    # Process the labels
    processed = processor.process_image_labels(image_labels, normalize=True)
    
    print("\nProcessed Labels:")
    for i, label in enumerate(processed, 1):
        print(f"  {i}. {label}")
    
    # Extract keywords
    all_text = ' '.join(image_labels)
    keywords = processor.extract_keywords(all_text, top_n=8)
    
    print("\n🔑 Top Keywords:")
    for word, freq in keywords:
        print(f"  '{word}': appears {freq} times")


def example_3_class_categories():
    """Example 3: Categorizing YOLO class names"""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: YOLO Class Categorization")
    print("=" * 50)
    
    # Common YOLO class names
    yolo_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "bird", "cat", "dog", "horse", "sheep", "cow",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"
    ]
    
    print(f"Analyzing {len(yolo_classes)} YOLO class names...")
    
    # Create category mapping
    category_mapping = create_word_based_class_mapping(yolo_classes)
    
    print("\n📂 Class Categories:")
    for category, indices in category_mapping.items():
        if indices:
            class_names_in_category = [yolo_classes[i] for i in indices]
            print(f"  {category.title()}: {class_names_in_category}")


def example_4_text_augmentation():
    """Example 4: Text augmentation for data diversity"""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Text Augmentation")
    print("=" * 50)
    
    # Sample descriptions
    original_texts = [
        "person walking with dog",
        "red car on street",
        "bird flying in sky"
    ]
    
    print("Text Augmentation Examples:")
    
    for text in original_texts:
        print(f"\n📝 Original: '{text}'")
        
        # Generate variations
        augmentations = generate_text_augmentations(text, num_augmentations=3)
        
        for i, aug in enumerate(augmentations[1:], 1):  # Skip original
            print(f"   Variation {i}: '{aug}'")


def example_5_text_quality():
    """Example 5: Text quality validation"""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Text Quality Validation")
    print("=" * 50)
    
    processor = TextProcessor()
    
    # Test different quality texts
    test_texts = [
        "A good quality description of the image",  # Good
        "a",  # Too short
        "THIS IS ALL CAPS!!!",  # Issues
        "the the the same word repeated repeatedly",  # Repetitive
        ""  # Empty
    ]
    
    print("Text Quality Assessment:")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")
        
        if text:  # Skip empty text for validation
            validation = processor.validate_text_quality(text)
            
            # Show validation results
            issues = [key for key, value in validation.items() if not value]
            if issues:
                print(f"   ❌ Issues: {', '.join(issues)}")
                
                # Get suggestions
                suggestions = processor.suggest_text_improvements(text)
                if suggestions:
                    print(f"   💡 Suggestions: {suggestions[0]}")
            else:
                print("   ✅ Good quality text")
        else:
            print("   ❌ Empty text")


def example_6_practical_integration():
    """Example 6: Practical integration with detection results"""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: Integration with Detection Results")
    print("=" * 50)
    
    # Simulate detection results
    detection_results = [
        {"class": 0, "name": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
        {"class": 2, "name": "car", "confidence": 0.87, "bbox": [300, 150, 500, 250]},
        {"class": 16, "name": "dog", "confidence": 0.92, "bbox": [50, 200, 150, 280]},
        {"class": 5, "name": "bus", "confidence": 0.78, "bbox": [400, 100, 600, 300]}
    ]
    
    print("Detection Results:")
    for result in detection_results:
        print(f"  {result['name']}: {result['confidence']:.2f} confidence")
    
    # Analyze detected object names
    processor = TextProcessor()
    object_names = [result['name'] for result in detection_results]
    
    # Get keywords from detected objects
    keywords = processor.extract_keywords(' '.join(object_names))
    print(f"\n🔍 Scene Keywords: {list(keywords.keys())}")
    
    # Generate scene description
    summary = processor.generate_text_summary(object_names, max_words=10)
    print(f"📝 Scene Summary: {summary}")


def main():
    """Run all examples"""
    print("🚀 YOLOv5 Word Specifier - Basic Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_analysis()
        example_2_image_labels()
        example_3_class_categories()
        example_4_text_augmentation()
        example_5_text_quality()
        example_6_practical_integration()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("  1. Try modifying the examples with your own text")
        print("  2. Run the full demo: python examples/word_specifier_demo.py")
        print("  3. Check the documentation: docs/word_specifier_guide.md")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you're running from the YOLOv5 root directory")


if __name__ == "__main__":
    main()