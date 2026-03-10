"""
Test script to debug video processing issues
Run this to see detailed error messages
"""

import sys
import os

def test_video_processing(video_path):
    """Test video processing with detailed error reporting"""
    
    print("="*60)
    print("VIDEO PROCESSING DEBUG TEST")
    print("="*60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check if video file exists
    print(f"\n1. Checking video file...")
    print(f"   Path: {video_path}")
    if not os.path.exists(video_path):
        print(f"   ✗ ERROR: File does not exist!")
        return False
    else:
        print(f"   ✓ File exists")
        print(f"   Size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Check if model exists
    print(f"\n2. Checking model file...")
    model_path = r"C:\Users\Randall Chiang\Documents\GitHub\FYP---Academic-Emotion-Recognition-using-facial-expressions\all_emotions_final.h5"
    if not os.path.exists(model_path):
        print(f"   ✗ ERROR: Model file '{model_path}' not found!")
        print(f"   Make sure Multi_best.h5 is in the current directory")
        return False
    else:
        print(f"   ✓ Model file exists")
        print(f"   Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    # Test imports
    print(f"\n3. Testing imports...")
    try:
        import cv2
        print(f"   ✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"   ✗ OpenCV import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"   ✓ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"   ✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"   ✗ NumPy import failed: {e}")
        return False
    
    # Test video opening
    print(f"\n4. Testing video file opening...")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"   ✗ Cannot open video file")
            print(f"   Try converting to MP4 format")
            return False
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   ✓ Video opened successfully")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.2f} seconds")
        
        # Test reading a frame
        ret, frame = cap.read()
        if not ret:
            print(f"   ✗ Cannot read frames from video")
            return False
        print(f"   ✓ Can read frames")
        
        cap.release()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test loading model
    print(f"\n5. Testing model loading...")
    try:
        from emotion_recognition_app import AcademicEmotionRecognizer
        print(f"   Loading recognizer...")
        recognizer = AcademicEmotionRecognizer(model_path)
        print(f"   ✓ Recognizer loaded successfully")
        print(f"   ✓ ResNet50 loaded")
        print(f"   ✓ GRU model loaded")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Actually try processing
    print(f"\n6. Attempting to process video...")
    try:
        from emotion_recognition_app import process_video_file
        print(f"   This may take a while...")
        process_video_file(video_path, model_path)
        print(f"\n   ✓ SUCCESS! Video processed successfully")
        return True
        
    except Exception as e:
        print(f"\n   ✗ ERROR during processing:")
        print(f"   {str(e)}")
        print(f"\n   Full traceback:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_processing.py <video_file.mp4>")
        print("\nExample:")
        print("  python test_video_processing.py myvideo.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    success = test_video_processing(video_path)
    
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED")
        print("Video processing works correctly!")
    else:
        print("✗ TESTS FAILED")
        print("See errors above for details")
    print("="*60)
