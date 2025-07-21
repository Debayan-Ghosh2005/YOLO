#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLOv5 Text Integration Example

This script demonstrates how to integrate the word specifier system
with actual YOLOv5 detection and training workflows.

Usage:
    python examples/yolo_text_integration.py
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
from utils.word_utils import (
    create_word_based_class_mapping,
    filter_predictions_by_words,
    enhance_class_names_with_synonyms,
    export_word_analysis_report
)


def load_coco_classes():
    """Load COCO class names"""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]


def example_class_analysis():
    """Analyze COCO class names with word specifier"""
    print("🔍 COCO Class Analysis")
    print("-" * 40)
    
    coco_classes = load_coco_classes()
    
    # Create category mapping
    category_mapping = create_word_based_class_mapping(coco_classes)
    
    print("Class Categories:")
    for category, indices in category_mapping.items():
        if indices:
            classes = [coco_classes[i] for i in indices[:5]]  # Show first 5
            more = f" (+{len(indices)-5} more)" if len(indices) > 5 else ""
            print(f"  {category.title()}: {classes}{more}")
    
    return category_mapping


def example_detection_filtering():
    """Filter detection results by word categories"""
    print("\n🎯 Detection Result Filtering")
    print("-" * 40)
    
    # Simulate detection results
    detections = [
        {'class': 0, 'confidence': 0.95, 'name': 'person', 'bbox': [100, 100, 200, 300]},
        {'class': 2, 'confidence': 0.87, 'name': 'car', 'bbox': [300, 150, 500, 250]},
        {'class': 5, 'confidence': 0.78, 'name': 'bus', 'bbox': [400, 100, 600, 300]},
        {'class': 15, 'confidence': 0.92, 'name': 'cat', 'bbox': [50, 200, 150, 280]},
        {'class': 16, 'confidence': 0.88, 'name': 'dog', 'bbox': [200, 250, 300, 350]},
        {'class': 39, 'confidence': 0.75, 'name': 'bottle', 'bbox': [150, 300, 180, 400]}
    ]
    
    coco_classes = load_coco_classes()
    
    print("All Detections:")
    for det in detections:
        print(f"  {det['name']}: {det['confidence']:.2f}")
    
    # Filter for vehicles only
    vehicle_detections = filter_predictions_by_words(
        detections, 
        word_categories=['vehicles'], 
        class_names=coco_classes
    )
    
    print("\nVehicle Detections Only:")
    for det in vehicle_detections:
        print(f"  {det['name']}: {det['confidence']:.2f}")
    
    return detections


def example_scene_description():
    """Generate scene descriptions from detections"""
    print("\n📝 Scene Description Generation")
    print("-" * 40)
    
    processor = TextProcessor()
    
    # Sample detection results
    detected_objects = ['person', 'car', 'dog', 'tree', 'building']
    
    print(f"Detected objects: {detected_objects}")
    
    # Generate scene summary
    scene_summary = processor.generate_text_summary(detected_objects, max_words=8)
    print(f"Scene summary: {scene_summary}")
    
    # Analyze the scene
    scene_text = ' '.join(detected_objects)
    keywords = processor.extract_keywords(scene_text, top_n=3)
    
    print("Key scene elements:")
    for word, freq in keywords:
        print(f"  {word}")


def example_annotation_processing():
    """Process image annotations for training"""
    print("\n📋 Annotation Processing")
    print("-" * 40)
    
    processor = TextProcessor()
    
    # Sample image annotations/captions
    annotations = [
        "A person walking their dog in the park on a sunny day",
        "Red car parked next to a blue truck in the parking lot",
        "Children playing soccer while adults watch from benches",
        "Traffic light showing green signal at busy intersection",
        "Cat sitting on windowsill looking outside at birds",
        "Person riding bicycle on bike path through the forest"
    ]
    
    print("Sample Annotations:")
    for i, ann in enumerate(annotations[:3], 1):
        print(f"  {i}. {ann}")
    print(f"  ... and {len(annotations)-3} more")
    
    # Analyze the dataset
    analysis = processor.analyze_dataset_text(annotations)
    
    print(f"\nDataset Analysis:")
    print(f"  📊 Total samples: {analysis['num_samples']}")
    print(f"  📏 Average length: {analysis['avg_text_length']:.1f} words")
    print(f"  🔤 Unique words: {analysis['unique_words']}")
    
    # Process annotations for consistency
    processed = processor.batch_process_texts(annotations, operations=['normalize', 'clean'])
    
    print(f"\nProcessed {len(processed)} annotations for training consistency")
    
    return annotations


def example_class_enhancement():
    """Enhance class names with synonyms"""
    print("\n🔄 Class Name Enhancement")
    print("-" * 40)
    
    # Sample class names
    class_names = ['person', 'car', 'dog', 'cat', 'bicycle']
    
    print("Original class names:")
    for name in class_names:
        print(f"  {name}")
    
    # Enhance with synonyms
    enhanced = enhance_class_names_with_synonyms(class_names)
    
    print("\nEnhanced with synonyms:")
    for i, synonyms in enhanced.items():
        print(f"  {class_names[i]}: {synonyms}")


def example_quality_validation():
    """Validate annotation quality"""
    print("\n✅ Annotation Quality Validation")
    print("-" * 40)
    
    processor = TextProcessor()
    
    # Sample annotations with various quality issues
    test_annotations = [
        "A clear description of the image content",  # Good
        "car",  # Too short
        "THE PERSON IS WALKING!!!",  # All caps, excessive punctuation
        "person person person walking walking",  # Repetitive
        "",  # Empty
        "A very detailed and comprehensive description of what is happening in this particular image"  # Too long
    ]
    
    print("Quality Validation Results:")
    
    for i, annotation in enumerate(test_annotations, 1):
        print(f"\n{i}. '{annotation}'")
        
        if annotation.strip():
            validation = processor.validate_text_quality(annotation)
            suggestions = processor.suggest_text_improvements(annotation)
            
            # Check if there are any issues
            issues = [key for key, value in validation.items() if not value]
            
            if issues:
                print(f"   ❌ Issues: {', '.join(issues)}")
                if suggestions:
                    print(f"   💡 Suggestion: {suggestions[0]}")
            else:
                print("   ✅ Good quality")
        else:
            print("   ❌ Empty annotation")


def example_export_analysis():
    """Export comprehensive analysis report"""
    print("\n📊 Export Analysis Report")
    print("-" * 40)
    
    # Sample dataset texts
    dataset_texts = [
        "Person walking with dog in park",
        "Red car driving on highway",
        "Cat sitting on windowsill",
        "Children playing soccer",
        "Bird flying in blue sky",
        "Traffic light at intersection",
        "Person riding bicycle",
        "Dog running on beach"
    ]
    
    print(f"Exporting analysis for {len(dataset_texts)} samples...")
    
    # Export comprehensive report
    try:
        export_word_analysis_report(
            dataset_texts, 
            'word_analysis_report.json'
        )
        print("✅ Report exported to 'word_analysis_report.json'")
        print("   This report contains detailed statistics about your dataset text")
    except Exception as e:
        print(f"❌ Error exporting report: {e}")


def main():
    """Run all integration examples"""
    print("🚀 YOLOv5 Word Specifier - Integration Examples")
    print("=" * 60)
    print("This demonstrates how to integrate word processing with YOLO workflows\n")
    
    try:
        # Run all examples
        example_class_analysis()
        example_detection_filtering()
        example_scene_description()
        example_annotation_processing()
        example_class_enhancement()
        example_quality_validation()
        example_export_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All integration examples completed!")
        print("\n🎯 Integration Ideas:")
        print("  • Use class categorization for organized training")
        print("  • Filter detections by semantic categories")
        print("  • Generate automatic scene descriptions")
        print("  • Validate annotation quality before training")
        print("  • Enhance class names for better model understanding")
        print("  • Create text-based data augmentation")
        
        print("\n📚 Next Steps:")
        print("  1. Integrate with your training pipeline")
        print("  2. Use for dataset preprocessing")
        print("  3. Apply to inference post-processing")
        print("  4. Create custom word categories for your domain")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you're running from the YOLOv5 root directory")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()