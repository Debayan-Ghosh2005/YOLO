# Word Specifier System - Usage Guide

The Word Specifier system provides comprehensive text processing and classification capabilities for YOLOv5. This guide shows you how to use it effectively.

## Quick Start

### 1. Basic Word Analysis

```python
from utils.word_specifier import WordSpecifier

# Initialize the word specifier
specifier = WordSpecifier()

# Analyze some text
text = "The YOLOv5 model detects cars, people, and animals with excellent accuracy"
analysis = specifier.analyze_text(text)

print(f"Total words: {analysis['total_words']}")
print(f"Unique words: {analysis['unique_words']}")
print(f"Word categories: {analysis['categorized_words']}")
```

### 2. Text Processing for Computer Vision

```python
from utils.text_processor import TextProcessor

# Initialize text processor
processor = TextProcessor()

# Process image captions/labels
labels = [
    "A red car driving on the highway",
    "Person walking with a dog in the park",
    "Multiple birds flying in the blue sky"
]

# Clean and normalize the labels
processed_labels = processor.process_image_labels(labels, normalize=True)
print("Processed labels:", processed_labels)

# Extract keywords
keywords = processor.extract_keywords(' '.join(labels), top_n=5)
print("Top keywords:", keywords)
```

### 3. Class Name Enhancement

```python
from utils.word_utils import enhance_class_names_with_synonyms

# COCO class names example
class_names = ["person", "bicycle", "car", "motorcycle", "airplane"]

# Get enhanced names with synonyms
enhanced = enhance_class_names_with_synonyms(class_names)
for i, synonyms in enhanced.items():
    print(f"{class_names[i]}: {synonyms}")
```

## Running the Examples

### Method 1: Run the Demo Script

```bash
# Navigate to your YOLOv5 directory
cd /path/to/yolov5

# Run the comprehensive demo
python examples/word_specifier_demo.py
```

### Method 2: Interactive Python Session

```python
# Start Python in your YOLOv5 directory
python

# Import and use the modules
from utils.word_specifier import WordSpecifier
from utils.text_processor import TextProcessor

# Your code here...
```

### Method 3: Integration with YOLOv5 Training

You can integrate the word specifier with your training pipeline:

```python
# In your training script
from utils.text_processor import TextProcessor

processor = TextProcessor()

# Process class names for better organization
class_names = ["person", "car", "truck", "bicycle"]
analysis = processor.analyze_dataset_text(class_names)
print("Class analysis:", analysis)
```

## Configuration

### Custom Word Categories

Create a custom configuration file:

```yaml
# data/custom_word_config.yaml
word_categories:
  vehicles:
    - car
    - truck
    - bicycle
    - motorcycle
    - bus
  animals:
    - dog
    - cat
    - bird
    - horse
  colors:
    - red
    - blue
    - green
    - yellow

stop_words:
  - the
  - and
  - or
  - but
```

Then use it:

```python
from utils.word_specifier import WordSpecifier

# Load with custom config
specifier = WordSpecifier('data/custom_word_config.yaml')
```

## Advanced Usage

### 1. Dataset Text Analysis

```python
from utils.word_utils import export_word_analysis_report

# Analyze your dataset annotations
annotations = [
    "Person riding bicycle on street",
    "Red car in parking lot",
    "Dog running in park"
]

# Export comprehensive analysis
export_word_analysis_report(
    annotations, 
    'dataset_analysis.json'
)
```

### 2. Text Quality Validation

```python
from utils.text_processor import TextProcessor

processor = TextProcessor()

# Validate text quality
text = "A person walking"
validation = processor.validate_text_quality(text)
print("Validation results:", validation)

# Get improvement suggestions
suggestions = processor.suggest_text_improvements(text)
print("Suggestions:", suggestions)
```

### 3. Text Augmentation

```python
from utils.word_utils import generate_text_augmentations

# Generate variations of text
original = "person walking with dog"
augmentations = generate_text_augmentations(original, num_augmentations=3)

for i, aug in enumerate(augmentations):
    print(f"Variation {i}: {aug}")
```

## Integration Examples

### With YOLO Detection Results

```python
from utils.word_utils import filter_predictions_by_words

# Example predictions
predictions = [
    {'class': 0, 'confidence': 0.9, 'name': 'person'},
    {'class': 2, 'confidence': 0.8, 'name': 'car'},
    {'class': 16, 'confidence': 0.7, 'name': 'dog'}
]

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog']

# Filter predictions to only include vehicles
vehicle_predictions = filter_predictions_by_words(
    predictions, 
    word_categories=['vehicles'], 
    class_names=class_names
)
```

### With Custom Datasets

```python
from utils.word_utils import load_class_names_with_words

# Load and analyze your custom class names
class_info = load_class_names_with_words('data/my_classes.yaml')

for class_id, info in class_info.items():
    print(f"Class {class_id}: {info['name']} (Category: {info['category']})")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the YOLOv5 root directory
2. **Missing Dependencies**: The system uses standard Python libraries (yaml, numpy, etc.)
3. **Configuration Issues**: Check your YAML files for proper formatting

### Performance Tips

1. **Large Datasets**: Process text in batches for better performance
2. **Memory Usage**: Use generators for very large text collections
3. **Custom Categories**: Define specific word categories for your domain

## API Reference

### WordSpecifier Class

- `analyze_text(text)`: Comprehensive text analysis
- `categorize_words(words)`: Categorize words into predefined categories
- `get_word_frequency(text)`: Calculate word frequencies
- `filter_words(words, categories)`: Filter words by categories
- `preprocess_text(text)`: Clean and normalize text

### TextProcessor Class

- `process_image_labels(labels)`: Process image labels/captions
- `extract_keywords(text)`: Extract important keywords
- `analyze_dataset_text(texts)`: Analyze entire dataset
- `validate_text_quality(text)`: Check text quality
- `suggest_text_improvements(text)`: Get improvement suggestions

### Utility Functions

- `create_word_based_class_mapping()`: Map classes to word categories
- `enhance_class_names_with_synonyms()`: Add synonyms to class names
- `generate_text_augmentations()`: Create text variations
- `export_word_analysis_report()`: Generate analysis reports

## Next Steps

1. Run the demo script to see all features in action
2. Experiment with your own text data
3. Integrate with your YOLOv5 training pipeline
4. Create custom word categories for your specific use case
5. Use the text quality validation for dataset cleaning

For more examples and advanced usage, check the example scripts in the `examples/` directory.