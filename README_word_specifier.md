# YOLOv5 Word Specifier System

A comprehensive text processing and classification system integrated with YOLOv5 for enhanced computer vision workflows.

## 🚀 Quick Start

### Option 1: Run the Quick Demo
```bash
python run_word_examples.py
```

### Option 2: Basic Usage
```python
from utils.word_specifier import WordSpecifier

# Initialize and analyze text
specifier = WordSpecifier()
analysis = specifier.analyze_text("YOLOv5 detects cars and people accurately")
print(analysis)
```

### Option 3: Process Image Labels
```python
from utils.text_processor import TextProcessor

processor = TextProcessor()
labels = ["person walking", "red car driving", "dog running"]
processed = processor.process_image_labels(labels, normalize=True)
print(processed)
```

## 📁 System Components

### Core Modules
- **`utils/word_specifier.py`** - Main word classification and analysis engine
- **`utils/text_processor.py`** - Advanced text processing for computer vision
- **`utils/word_utils.py`** - Utility functions for YOLO integration

### Configuration
- **`data/word_config.yaml`** - Word categories and stop words configuration

### Examples
- **`examples/word_specifier_demo.py`** - Comprehensive demonstration
- **`examples/basic_word_usage.py`** - Simple usage examples
- **`examples/yolo_text_integration.py`** - Integration with YOLO workflows
- **`run_word_examples.py`** - Quick start script

### Documentation
- **`docs/word_specifier_guide.md`** - Complete usage guide

## 🎯 Key Features

### 1. Text Analysis
- Word frequency analysis
- Text categorization
- Quality validation
- Statistical analysis

### 2. YOLO Integration
- Class name categorization
- Detection result filtering
- Scene description generation
- Annotation processing

### 3. Text Processing
- Normalization and cleaning
- Keyword extraction
- Text augmentation
- Quality suggestions

### 4. Customization
- Custom word categories
- Configurable stop words
- Domain-specific processing
- Extensible architecture

## 📊 Use Cases

### Computer Vision Applications
- **Dataset Analysis**: Analyze and categorize your training data
- **Quality Control**: Validate annotation quality before training
- **Scene Understanding**: Generate descriptions from detection results
- **Data Augmentation**: Create text variations for robust training

### Text Processing Tasks
- **Label Cleaning**: Normalize and standardize image labels
- **Keyword Extraction**: Find important terms in descriptions
- **Category Mapping**: Organize classes by semantic meaning
- **Content Filtering**: Filter results by text categories

## 🛠️ Installation & Setup

The word specifier system uses standard Python libraries included with YOLOv5:
- `yaml` - Configuration file handling
- `numpy` - Numerical operations
- `pathlib` - File path operations

No additional installation required!

## 📖 Examples

### Analyze COCO Classes
```python
from utils.word_utils import create_word_based_class_mapping

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus"]
categories = create_word_based_class_mapping(coco_classes)
print(categories)
# Output: {'vehicles': [1, 2, 3, 4, 5], 'people': [0], ...}
```

### Filter Detections by Category
```python
from utils.word_utils import filter_predictions_by_words

predictions = [
    {'class': 0, 'name': 'person', 'confidence': 0.9},
    {'class': 2, 'name': 'car', 'confidence': 0.8}
]

# Keep only vehicles
vehicles = filter_predictions_by_words(
    predictions, 
    word_categories=['vehicles'], 
    class_names=coco_classes
)
```

### Generate Text Variations
```python
from utils.word_utils import generate_text_augmentations

original = "person walking with dog"
variations = generate_text_augmentations(original, num_augmentations=3)
print(variations)
# Output: ['person walking with dog', 'individual strolling with canine', ...]
```

### Validate Text Quality
```python
from utils.text_processor import TextProcessor

processor = TextProcessor()
validation = processor.validate_text_quality("A good description")
suggestions = processor.suggest_text_improvements("bad text")
```

## 🔧 Configuration

### Custom Word Categories
Edit `data/word_config.yaml`:

```yaml
word_categories:
  vehicles:
    - car
    - truck
    - bicycle
    - motorcycle
  animals:
    - dog
    - cat
    - bird
    - horse
  colors:
    - red
    - blue
    - green
```

### Using Custom Config
```python
specifier = WordSpecifier('data/custom_config.yaml')
```

## 🎮 Running Examples

### All Examples at Once
```bash
# Quick demo
python run_word_examples.py

# Basic usage examples
python examples/basic_word_usage.py

# Full feature demonstration
python examples/word_specifier_demo.py

# YOLO integration examples
python examples/yolo_text_integration.py
```

### Interactive Usage
```python
# Start Python in YOLOv5 directory
python

# Import and use
from utils.word_specifier import WordSpecifier
from utils.text_processor import TextProcessor

specifier = WordSpecifier()
processor = TextProcessor()

# Your code here...
```

## 📈 Performance Tips

1. **Batch Processing**: Process multiple texts together for better performance
2. **Custom Categories**: Define specific categories for your domain
3. **Memory Management**: Use generators for large datasets
4. **Caching**: Results can be cached for repeated analysis

## 🤝 Integration with YOLOv5

### Training Pipeline
```python
# In your training script
from utils.text_processor import TextProcessor

processor = TextProcessor()

# Analyze class names
class_analysis = processor.analyze_dataset_text(class_names)
print(f"Dataset has {class_analysis['unique_words']} unique concepts")
```

### Inference Pipeline
```python
# After detection
from utils.word_utils import filter_predictions_by_words

# Filter results by category
filtered_results = filter_predictions_by_words(
    predictions, 
    word_categories=['vehicles'], 
    class_names=model.names
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the YOLOv5 root directory
   cd /path/to/yolov5
   python run_word_examples.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install pyyaml numpy
   ```

3. **Configuration Issues**
   - Check YAML file formatting
   - Ensure file paths are correct
   - Validate configuration syntax

### Getting Help

1. Run the quick demo: `python run_word_examples.py`
2. Check the examples in `examples/` directory
3. Read the full guide: `docs/word_specifier_guide.md`
4. Look at the configuration: `data/word_config.yaml`

## 🎯 Next Steps

1. **Start Simple**: Run `python run_word_examples.py`
2. **Explore Examples**: Try different example scripts
3. **Customize**: Edit `data/word_config.yaml` for your needs
4. **Integrate**: Add to your YOLOv5 training/inference pipeline
5. **Extend**: Create custom categories and processing rules

---

**Ready to enhance your YOLOv5 project with powerful text processing capabilities!** 🚀