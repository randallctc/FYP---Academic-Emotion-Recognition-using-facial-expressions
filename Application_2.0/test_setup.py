"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow (PIL) installed")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import tkinter as tk
        print(f"✓ Tkinter available")
    except ImportError as e:
        print(f"✗ Tkinter import failed: {e}")
        print("  Note: Tkinter usually comes with Python, but may need separate installation on some systems")
        return False
    
    return True


def test_resnet50():
    """Test if ResNet50 can be loaded"""
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50
    
    print("\nTesting ResNet50 feature extractor...")
    try:
        # Try to load ResNet50
        print("  Loading ResNet50 (this may download ~100MB on first run)...")
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        print(f"✓ ResNet50 loaded successfully")
        print(f"  Output shape: {model.output_shape}")
        
        # Test prediction
        import numpy as np
        test_input = np.random.random((1, 224, 224, 3)).astype('float32') * 255
        features = model.predict(test_input, verbose=0)
        
        if features.shape == (1, 2048):
            print(f"✓ ResNet50 feature extraction working (output: {features.shape})")
            return True
        else:
            print(f"✗ Unexpected ResNet50 output shape: {features.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load ResNet50: {e}")
        print("  This is required for feature extraction!")
        return False


def test_camera():
    """Test if camera is accessible"""
    import cv2
    
    print("\nTesting camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Cannot access camera (ID: 0)")
        print("  Try different camera IDs or check camera permissions")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"✓ Camera accessible (frame shape: {frame.shape})")
        return True
    else:
        print("✗ Camera opened but cannot read frames")
        return False


def test_face_detection():
    """Test if Haar Cascade is available"""
    import cv2
    
    print("\nTesting face detection...")
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("✗ Haar Cascade file is empty")
            return False
        print("✓ Face detection cascade loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load face detection: {e}")
        return False


def main():
    print("=" * 60)
    print("Academic Emotion Recognition - Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test ResNet50
    if not test_resnet50():
        all_passed = False
    
    # Test camera
    if not test_camera():
        all_passed = False
    
    # Test face detection
    if not test_face_detection():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to go.")
        print("\nNext steps:")
        print("1. Place your Multi_best.h5 model (GRU) in this directory")
        print("   • Your model should accept (batch, 60, 2048) ResNet50 features")
        print("   • ResNet50 feature extraction is handled automatically")
        print("2. Run: python emotion_recognition_gui.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
