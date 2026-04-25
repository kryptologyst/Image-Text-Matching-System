"""Interactive demo application for image-text matching."""

import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.clip_model import CLIPImageTextMatcher
from src.utils.device import get_device
from src.viz.visualizer import ImageTextVisualizer


def load_model():
    """Load the CLIP model."""
    if "model" not in st.session_state:
        with st.spinner("Loading CLIP model..."):
            device = get_device("auto")
            model = CLIPImageTextMatcher()
            model.to(device)
            model.eval()
            processor = model.get_processor()
            
            st.session_state.model = model
            st.session_state.processor = processor
            st.session_state.device = device
            st.session_state.visualizer = ImageTextVisualizer(model, processor, device)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Image-Text Matching Demo",
        page_icon="🖼️",
        layout="wide",
    )
    
    st.title("🖼️ Image-Text Matching System")
    st.markdown("**A modern CLIP-based system for cross-modal retrieval and matching**")
    
    # Load model
    load_model()
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    similarity_metric = st.sidebar.selectbox(
        "Similarity Metric",
        ["cosine", "dot_product", "euclidean"],
        index=0,
    )
    
    top_k = st.sidebar.slider("Top-K Results", 1, 20, 5)
    
    # Safety settings
    st.sidebar.subheader("Safety Settings")
    safety_filters = st.sidebar.checkbox("Enable Safety Filters", value=True)
    content_warning = st.sidebar.checkbox("Show Content Warnings", value=True)
    
    if content_warning:
        st.sidebar.warning(
            "⚠️ **Content Warning**: This demo is for research and educational purposes only. "
            "Results may not be suitable for all audiences."
        )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Image-to-Text", 
        "📝 Text-to-Image", 
        "🔍 Similarity Matrix", 
        "📊 Attention Maps"
    ])
    
    with tab1:
        st.header("Image-to-Text Retrieval")
        st.markdown("Upload an image and find the most relevant text descriptions.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an image to find matching text descriptions"
        )
        
        # Sample texts
        sample_texts = [
            "A cute cat sitting on a windowsill",
            "A happy dog playing in the park", 
            "A beautiful mountain landscape at sunset",
            "A red sports car on a highway",
            "Delicious pasta with tomato sauce",
            "A modern city skyline at night",
            "A peaceful lake with swans",
            "A vintage bicycle on a cobblestone street",
            "A colorful flower garden in spring",
            "A cozy fireplace in a living room",
        ]
        
        # Text input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            custom_texts = st.text_area(
                "Enter text descriptions (one per line):",
                value="\n".join(sample_texts),
                height=200,
                help="Enter text descriptions to match against the uploaded image"
            )
        
        with col2:
            st.markdown("**Sample Texts:**")
            for i, text in enumerate(sample_texts[:5]):
                st.markdown(f"{i+1}. {text}")
        
        # Process texts
        texts = [line.strip() for line in custom_texts.split("\n") if line.strip()]
        
        if uploaded_file and texts:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Compute similarities
                with st.spinner("Computing similarities..."):
                    model = st.session_state.model
                    processor = st.session_state.processor
                    device = st.session_state.device
                    
                    # Get image embedding
                    with torch.no_grad():
                        inputs = processor(images=image, return_tensors="pt")
                        pixel_values = inputs["pixel_values"].to(device)
                        image_embed = model.encode_image(pixel_values)
                        
                        # Get text embeddings
                        text_embeds = []
                        for text in texts:
                            inputs = processor(text=[text], return_tensors="pt")
                            input_ids = inputs["input_ids"].to(device)
                            attention_mask = inputs["attention_mask"].to(device)
                            embed = model.encode_text(input_ids, attention_mask)
                            text_embeds.append(embed.cpu())
                        
                        # Compute similarities
                        text_embeds = torch.cat(text_embeds, dim=0)
                        similarities = torch.matmul(image_embed, text_embeds.t()).squeeze()
                        
                        # Get top-k results
                        top_k_indices = torch.topk(similarities, min(top_k, len(texts))).indices
                
                # Display results
                st.markdown("### 🎯 Retrieval Results")
                
                for i, idx in enumerate(top_k_indices):
                    similarity_score = similarities[idx].item()
                    
                    # Color coding based on similarity
                    if similarity_score > 0.7:
                        color = "🟢"
                    elif similarity_score > 0.5:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.markdown(
                        f"**{i+1}.** {color} `{similarity_score:.3f}` - {texts[idx]}"
                    )
    
    with tab2:
        st.header("Text-to-Image Retrieval")
        st.markdown("Enter a text description and find the most relevant images.")
        
        # Text input
        query_text = st.text_input(
            "Enter a text description:",
            value="A cute cat sitting on a windowsill",
            help="Enter a text description to find matching images"
        )
        
        # Sample images (placeholder)
        st.markdown("### Sample Images")
        st.info("📝 **Note**: In a real implementation, you would have a database of images to search through. "
                "This demo shows the interface for text-to-image retrieval.")
        
        if query_text:
            st.markdown(f"**Query:** {query_text}")
            st.markdown("**Top matching images would appear here:**")
            
            # Placeholder for image results
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    st.image(
                        "https://via.placeholder.com/200x200?text=Sample+Image",
                        caption=f"Rank {i+1}: Similarity 0.8{i}",
                        use_column_width=True
                    )
    
    with tab3:
        st.header("Similarity Matrix Visualization")
        st.markdown("Visualize similarities between multiple images and texts.")
        
        # Upload multiple images
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            help="Upload multiple images to create a similarity matrix"
        )
        
        # Text inputs
        matrix_texts = st.text_area(
            "Enter text descriptions for the matrix:",
            value="A cat\nA dog\nA car\nA house\nA tree",
            height=150,
            help="Enter text descriptions (one per line)"
        )
        
        texts_for_matrix = [line.strip() for line in matrix_texts.split("\n") if line.strip()]
        
        if uploaded_files and texts_for_matrix:
            st.markdown("### Similarity Matrix")
            
            # Process images
            images = [Image.open(file).convert("RGB") for file in uploaded_files]
            
            # Limit to reasonable number for visualization
            max_items = 5
            if len(images) > max_items:
                st.warning(f"Showing only the first {max_items} images for visualization")
                images = images[:max_items]
            
            if len(texts_for_matrix) > max_items:
                st.warning(f"Showing only the first {max_items} texts for visualization")
                texts_for_matrix = texts_for_matrix[:max_items]
            
            # Create similarity matrix visualization
            try:
                visualizer = st.session_state.visualizer
                fig = visualizer.visualize_similarity_matrix(
                    images, texts_for_matrix, figsize=(8, 6)
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating similarity matrix: {str(e)}")
    
    with tab4:
        st.header("Attention Maps Visualization")
        st.markdown("Visualize attention patterns between images and text.")
        
        # Image upload
        attention_image = st.file_uploader(
            "Choose an image for attention visualization",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="attention_image",
            help="Upload an image to visualize attention maps"
        )
        
        # Text input
        attention_text = st.text_input(
            "Enter text for attention visualization:",
            value="A cute cat sitting on a windowsill",
            help="Enter text to visualize attention patterns"
        )
        
        if attention_image and attention_text:
            image = Image.open(attention_image).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Input Image", use_column_width=True)
                st.markdown(f"**Text:** {attention_text}")
            
            with col2:
                try:
                    visualizer = st.session_state.visualizer
                    fig = visualizer.visualize_attention_maps(
                        image, attention_text, figsize=(12, 4)
                    )
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating attention maps: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer**: This demo is for research and educational purposes only. "
        "The model may not always produce accurate or appropriate results. "
        "Please use responsibly and consider the limitations of AI systems."
    )


if __name__ == "__main__":
    main()
