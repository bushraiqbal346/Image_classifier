# =============================================================================
# STEP 1: INSTALL REQUIRED PACKAGES
# =============================================================================
# Run this cell first to install all necessary packages

!pip install streamlit torch transformers opencv-python pillow matplotlib numpy accelerate

print("✅ All packages installed successfully!")

# =============================================================================
# STEP 2: IMPORT LIBRARIES
# =============================================================================
# Run this cell to import all required libraries

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import io
import threading
import time

print("✅ All libraries imported successfully!")

# =============================================================================
# STEP 3: DEFINE HELPER FUNCTIONS
# =============================================================================
# These functions will help us analyze images

def load_models():
    """Load AI models for object detection and image captioning"""
    print("🔄 Loading AI models... This may take 2-3 minutes for the first time.")
    
    try:
        # Object Detection Model (DETR)
        print("Loading object detection model...")
        detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Image Captioning Model (BLIP)
        print("Loading image captioning model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        print("✅ All models loaded successfully!")
        return detr_processor, detr_model, blip_processor, blip_model
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return None, None, None, None

def detect_objects(image, processor, model, threshold=0.7):
    """Detect objects in the image"""
    try:
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt")
        
        # Get predictions from the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process the results
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            detections.append({
                "label": label_name,
                "confidence": round(score.item(), 3),
                "box": box
            })
        
        return detections
    except Exception as e:
        print(f"❌ Error in object detection: {str(e)}")
        return []

def generate_caption(image, processor, model):
    """Generate a description of what's happening in the image"""
    try:
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"❌ Error generating caption: {str(e)}")
        return "Unable to generate caption"

def count_objects(detections):
    """Count how many of each object type we found"""
    object_counts = Counter([det["label"] for det in detections])
    return dict(object_counts)

def visualize_detections(image, detections):
    """Create a visualization showing where objects are detected"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Use different colors for each detection
    colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
    
    for detection, color in zip(detections, colors):
        box = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        # Draw bounding box around the object
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with confidence score
        ax.text(
            box[0], box[1] - 10,
            f"{label} ({confidence:.2f})",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            fontsize=10,
            color='black'
        )
    
    ax.set_title("Object Detection Results", fontsize=16)
    ax.axis('off')
    
    return fig

print("✅ Helper functions defined successfully!")

# =============================================================================
# STEP 4: LOAD THE AI MODELS
# =============================================================================
# Run this cell to load the AI models (this will take some time)

detr_processor, detr_model, blip_processor, blip_model = load_models()

# =============================================================================
# STEP 5: SIMPLE IMAGE ANALYSIS FUNCTION
# =============================================================================
# This function analyzes any image you provide

def analyze_image(image_path_or_url, confidence_threshold=0.7):
    """
    Analyze an image and return:
    1. Description of what's happening
    2. List of detected objects
    3. Count of each object type
    4. Visualization
    """
    print(f"🔍 Analyzing image...")
    
    # Load the image
    if image_path_or_url.startswith('http'):
        # If it's a URL, download the image
        import requests
        response = requests.get(image_path_or_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        # If it's a file path, open it directly
        image = Image.open(image_path_or_url).convert("RGB")
    
    print("✅ Image loaded successfully!")
    
    # Generate description
    print("📝 Generating image description...")
    description = generate_caption(image, blip_processor, blip_model)
    
    # Detect objects
    print("🔍 Detecting objects...")
    detections = detect_objects(image, detr_processor, detr_model, confidence_threshold)
    
    # Count objects
    object_counts = count_objects(detections)
    
    # Display results
    print("\n" + "="*50)
    print("🎯 ANALYSIS RESULTS")
    print("="*50)
    
    print(f"\n📖 IMAGE DESCRIPTION:")
    print(f"   {description}")
    
    print(f"\n🔢 OBJECT COUNTS:")
    if object_counts:
        for obj, count in object_counts.items():
            print(f"   • {obj}: {count}")
    else:
        print("   No objects detected with current confidence threshold")
    
    print(f"\n📋 DETAILED DETECTIONS:")
    if detections:
        for i, detection in enumerate(detections, 1):
            print(f"   {i}. {detection['label']} (Confidence: {detection['confidence']:.2f})")
    else:
        print("   No objects detected")
    
    # Show visualization
    if detections:
        print(f"\n🎨 Creating visualization...")
        fig = visualize_detections(image, detections)
        plt.show()
    
    # Show original image
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    return {
        'description': description,
        'detections': detections,
        'object_counts': object_counts,
        'total_objects': len(detections)
    }

print("✅ Image analysis function ready!")

# =============================================================================
# STEP 6: STREAMLIT WEB APP (OPTIONAL)
# =============================================================================
# This creates a web interface for the image classifier

def create_streamlit_app():
    """Create the Streamlit web app"""
    
    # Create the app file
    app_code = '''
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from transformers import DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import io

# Set page configuration
st.set_page_config(
    page_title="Image Classifier Agent",
    page_icon="🔍",
    layout="wide"
)

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    """Load all required models"""
    try:
        # Object Detection Model (DETR)
        detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Image Captioning Model (BLIP)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        return detr_processor, detr_model, blip_processor, blip_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def detect_objects(image, processor, model, threshold=0.7):
    """Detect objects in image using DETR model"""
    try:
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = model.config.id2label[label.item()]
            detections.append({
                "label": label_name,
                "confidence": round(score.item(), 3),
                "box": box
            })
        
        return detections
    except Exception as e:
        st.error(f"Error in object detection: {str(e)}")
        return []

def generate_caption(image, processor, model):
    """Generate caption for the image"""
    try:
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Unable to generate caption"

def count_objects(detections):
    """Count detected objects"""
    object_counts = Counter([det["label"] for det in detections])
    return dict(object_counts)

def visualize_detections(image, detections):
    """Visualize object detections on image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
    
    for detection, color in zip(detections, colors):
        box = detection["box"]
        label = detection["label"]
        confidence = detection["confidence"]
        
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor=color, 
            facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            box[0], box[1] - 10,
            f"{label} ({confidence:.2f})",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
            fontsize=10,
            color='black'
        )
    
    ax.set_title("Object Detection Results", fontsize=16)
    ax.axis('off')
    
    return fig

def main():
    st.title("🔍 Image Classifier Agent")
    st.markdown("Upload an image to detect objects, get descriptions, and count items!")
    
    # Load models
    with st.spinner("Loading AI models... This may take a moment."):
        detr_processor, detr_model, blip_processor, blip_model = load_models()
    
    if not all([detr_processor, detr_model, blip_processor, blip_model]):
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.1
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Generate caption
            with st.spinner("Analyzing image..."):
                caption = generate_caption(image, blip_processor, blip_model)
                st.write("**Image Description:**")
                st.write(caption)
            
            # Detect objects
            with st.spinner("Detecting objects..."):
                detections = detect_objects(image, detr_processor, detr_model, confidence_threshold)
                
                if detections:
                    # Count objects
                    object_counts = count_objects(detections)
                    
                    st.write("**Object Counts:**")
                    for obj, count in object_counts.items():
                        st.write(f"- {obj}: {count}")
                    
                    # Show detailed detections
                    st.write("**Detailed Detections:**")
                    for i, detection in enumerate(detections, 1):
                        st.write(f"{i}. {detection['label']} (Confidence: {detection['confidence']:.2f})")
                else:
                    st.write("No objects detected with the current confidence threshold.")
        
        # Visualization
        if detections:
            st.subheader("Object Detection Visualization")
            with st.spinner("Creating visualization..."):
                fig = visualize_detections(image, detections)
                st.pyplot(fig)
        
        # Summary statistics
        if detections:
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Objects", len(detections))
            
            with col2:
                st.metric("Unique Object Types", len(set(det["label"] for det in detections)))
            
            with col3:
                avg_confidence = sum(det["confidence"] for det in detections) / len(detections)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")

if __name__ == "__main__":
    main()
'''
    
    # Save the app code to a file
    with open('streamlit_app.py', 'w') as f:
        f.write(app_code)
    
    print("✅ Streamlit app created as 'streamlit_app.py'")
    print("📝 To run the web app, execute: !streamlit run streamlit_app.py &")

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

print("\n" + "="*60)
print("🚀 SETUP COMPLETE! HERE'S HOW TO USE THE IMAGE CLASSIFIER:")
print("="*60)

print("""
METHOD 1: ANALYZE IMAGE DIRECTLY (SIMPLE)
-----------------------------------------
# Upload an image file to Colab or use a URL
# Then run:
results = analyze_image('path_to_your_image.jpg')

# Or use a URL:
results = analyze_image('https://example.com/image.jpg')

METHOD 2: CREATE WEB APP (ADVANCED)
-----------------------------------
# Run this to create a Streamlit web app:
create_streamlit_app()

# Then run the web app:
!streamlit run streamlit_app.py &

# You'll get a URL to access your web app!

EXAMPLE USAGE:
--------------
# Let's try with a sample image URL:
sample_url = "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
results = analyze_image(sample_url)
""")

print("\n🎉 Your Image Classifier Agent is ready to use!")
print("📝 Copy and paste the example code above to start analyzing images!")
