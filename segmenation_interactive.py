import streamlit as st
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import torch

# Initialize SAM Model
def load_sam_model():
    sam_checkpoint = "/Users/pratyushranjan/Desktop/OE/sam_vit_h_4b8939.pth"  # Path to the SAM checkpoint
    model_type = "vit_h"  # Type of SAM model: vit_h, vit_l, vit_b

    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return SamPredictor(sam)

# Segment image based on user inputs
def segment_image(predictor, image, input_points, input_labels):
    # Set the input image
    predictor.set_image(image)

    # Predict masks based on user inputs
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    return masks, scores

# Convert masks to overlay on the original image
def overlay_masks(image, masks):
    overlay = image.copy()
    for mask in masks:
        color = np.random.randint(0, 255, size=3)  # Random color for each mask
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    return overlay

# Streamlit app
def main():
    st.title("Interactive Image Segmentation with Segment Anything Model")

    # Sidebar
    st.sidebar.header("Instructions")
    st.sidebar.write(
        "1. Upload an image.\n"
        "2. Click on the image to add points for segmentation.\n"
        "3. View the segmented regions."
    )

    # Load SAM model
    predictor = load_sam_model()

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load image
        image = np.array(Image.open(uploaded_file).convert("RGB"))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Point selection via clicks
        st.write("Click on the image to add points for segmentation.")
        input_points = st.session_state.get("input_points", [])
        input_labels = st.session_state.get("input_labels", [])

        if "input_points" not in st.session_state:
            st.session_state.input_points = []
            st.session_state.input_labels = []

        # Select points on the image
        selected_point = st.image(image, caption="Click to select points", use_column_width=True)
        clicked = st.button("Add Point")

        if clicked:
            x = st.number_input("X Coordinate", min_value=0, max_value=image.shape[1]-1)
            y = st.number_input("Y Coordinate", min_value=0, max_value=image.shape[0]-1)
            label = st.selectbox("Label (1 for foreground, 0 for background)", [1, 0])

            st.session_state.input_points.append([x, y])
            st.session_state.input_labels.append(label)

        # Display selected points
        st.write("Selected Points:", st.session_state.input_points)

        if st.button("Segment Image"):
            with st.spinner("Processing the segmentation, please wait..."):
                # Perform segmentation
                input_points = np.array(st.session_state.input_points)
                input_labels = np.array(st.session_state.input_labels)
                masks, scores = segment_image(predictor, image, input_points, input_labels)

                # Overlay masks
                overlayed_image = overlay_masks(image, masks)
                st.image(overlayed_image, caption="Segmented Image", use_column_width=True)

if __name__ == "__main__":
    main()
