#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Quick Start Script for Word Specifier System

This is the easiest way to get started with the word specifier system.
Just run this script to see everything in action!

Usage:
    python run_word_examples.py
"""

import sys
from pathlib import Path

# Ensure we can import from utils
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import yaml
        import numpy as np
        print("✅ Dependencies check passed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("  pip install pyyaml numpy")
        return False

def quick_demo():
    """Quick demonstration of word specifier capabilities"""
    print("🚀 YOLOv5 Word Specifier - Quick Demo")
    print("=" * 50)
    
    try:
        from utils.word_specifier import WordSpecifier
        from utils.text_processor import TextProcessor
        
        # Initialize
        specifier = WordSpecifier()
        processor = TextProcessor()
        
        print("✅ Word Specifier system loaded successfully!")
        
        # Quick text analysis
        sample_text = "YOLOv5 detects cars, people, and animals with excellent accuracy"
        analysis = specifier.analyze_text(sample_text)
        
        print(f"\n📊 Quick Analysis of: '{sample_text}'")
        print(f"   Words: {analysis['total_words']}")
        print(f"   Categories found: {len([k for k, v in analysis['categorized_words'].items() if v])}")
        
        # Quick label processing
        labels = ["person walking", "red car driving", "dog running"]
        processed = processor.process_image_labels(labels)
        
        print(f"\n🏷️  Processed {len(labels)} labels:")
        for orig, proc in zip(labels, processed):
            print(f"   '{orig}' → '{proc}'")
        
        print("\n🎉 Quick demo completed! The system is working correctly.")
        print("\nTo see more examples, run:")
        print("  python examples/basic_word_usage.py")
        print("  python examples/word_specifier_demo.py")
        print("  python examples/yolo_text_integration.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the YOLOv5 root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_usage_options():
    """Show different ways to use the word specifier system"""
    print("\n" + "=" * 50)
    print("📖 USAGE OPTIONS")
    print("=" * 50)
    
    print("\n1️⃣  Quick Start (you just ran this):")
    print("   python run_word_examples.py")
    
    print("\n2️⃣  Basic Examples:")
    print("   python examples/basic_word_usage.py")
    
    print("\n3️⃣  Full Demo:")
    print("   python examples/word_specifier_demo.py")
    
    print("\n4️⃣  YOLO Integration:")
    print("   python examples/yolo_text_integration.py")
    
    print("\n5️⃣  Interactive Python:")
    print("   python")
    print("   >>> from utils.word_specifier import WordSpecifier")
    print("   >>> specifier = WordSpecifier()")
    print("   >>> specifier.analyze_text('your text here')")
    
    print("\n6️⃣  Custom Configuration:")
    print("   # Edit data/word_config.yaml")
    print("   # Then use: WordSpecifier('data/word_config.yaml')")
    
    print("\n📚 Documentation:")
    print("   Check docs/word_specifier_guide.md for detailed usage")

def main():
    """Main function"""
    if not check_dependencies():
        return
    
    quick_demo()
    show_usage_options()
    
    print("\n" + "=" * 50)
    print("🎯 Ready to use the Word Specifier system!")
    print("=" * 50)

if __name__ == "__main__":
    main()