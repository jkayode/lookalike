import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import hashlib
import json
import io
from datetime import datetime
import dropbox
import requests

# Replace these with your details
client_id = st.secrets["APP_KEY"]
client_secret = st.secrets["APP_SECRET"]
refresh_token = st.secrets["REFRESH_TOKEN"]

def refresh_access_token(client_id, client_secret, refresh_token):
    url = "https://api.dropbox.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to refresh token: {response.json()}")

# Example usage
new_access_token = refresh_access_token(client_id, client_secret, refresh_token)

# Dropbox setup
DROPBOX_ACCESS_TOKEN = new_access_token  # Replace with your Dropbox access token
DROPBOX_FOLDER = "/image_database"
METADATA_FILE = f"{DROPBOX_FOLDER}/image_metadata.json"
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Load MobileNetV2 model
mobilenet_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Helper functions for Dropbox operations
def load_metadata():
    try:
        _, res = dbx.files_download(METADATA_FILE)
        return json.load(res.raw)
    except dropbox.exceptions.ApiError:
        return {}

def save_metadata(metadata):
    dbx.files_upload(
        json.dumps(metadata).encode(),
        METADATA_FILE,
        mode=dropbox.files.WriteMode.overwrite
    )

def upload_to_dropbox(file, filename):
    file.seek(0)
    dropbox_path = f"{DROPBOX_FOLDER}/{filename}"
    dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    return dropbox_path

def download_from_dropbox(path):
    _, res = dbx.files_download(path)
    return Image.open(io.BytesIO(res.content))

# Load or initialize metadata
image_metadata = load_metadata()

# Function to preprocess images
def preprocess_image(image):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Function to compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Add images to Dropbox database
def add_to_database(files):
    for file in files:
        image = Image.open(file)
        img_array = preprocess_image(image)
        embedding = mobilenet_model.predict(img_array).flatten()

        image_key = hashlib.md5(file.getvalue()).hexdigest()
        file_name = f"{image_key}.jpg"

        if image_key not in image_metadata:
            dropbox_path = upload_to_dropbox(file, file_name)
            image_metadata[image_key] = {
                "name": file.name,
                "path": dropbox_path,
                "embedding": embedding.tolist(),
                "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.success(f"Added {file.name} to the database!")
        else:
            st.warning(f"{file.name} is already in the database.")
    save_metadata(image_metadata)

# Streamlit App
st.title("Visual Similarity Analyzer")
st.subheader("Lookalike Image Detection Tool")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Compare Images", "Batch Comparison", "Manage Database", "Algorithm Details"])

# Tab 1: Compare Images
with tab1:
    st.header("Compare Images")
    uploaded_new_file = st.file_uploader("Upload a New Image (JPEG, PNG)", type=["jpg", "jpeg", "png"])
    threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.01)

    if uploaded_new_file:
        st.subheader("Uploaded New Image")
        new_image = Image.open(uploaded_new_file)
        st.image(new_image, caption="New Image", width=300)

        new_embedding = mobilenet_model.predict(preprocess_image(new_image)).flatten()

        if st.button("Compare Against Database"):
            results = []
            for key, data in image_metadata.items():
                similarity = compute_cosine_similarity(new_embedding, np.array(data["embedding"]))
                if similarity >= threshold:
                    results.append({
                        "Database Image": data["name"],
                        "Similarity": similarity,
                        "Path": data["path"],
                    })

            if results:
                results_df = pd.DataFrame(results).sort_values(by="Similarity", ascending=False)
                st.write("Similarity Results", results_df)

                st.subheader("Visual Comparison")
                for _, row in results_df.iterrows():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(new_image, caption="New Image", width=200)
                    with col2:
                        db_image = download_from_dropbox(row["Path"])
                        st.image(db_image, caption=f"{row['Database Image']} (Similarity: {row['Similarity']:.2f})", width=200)
            else:
                st.warning("No similar images found above the threshold.")

# Tab 2: Batch Comparison
with tab2:
    st.header("Batch Image Comparison")
    batch_files = st.file_uploader(
        "Upload Multiple Images (JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    batch_threshold = st.slider("Batch Similarity Threshold", 0.5, 1.0, 0.8, 0.01, key="batch_slider")

    if st.button("Compare Batch Images"):
        if batch_files:
            embeddings = {}
            for file in batch_files:
                image = Image.open(file)
                embeddings[file.name] = {
                    "image": image,
                    "embedding": mobilenet_model.predict(preprocess_image(image)).flatten()
                }

            results = []
            image_names = list(embeddings.keys())
            for i, name1 in enumerate(image_names):
                for j, name2 in enumerate(image_names):
                    if i >= j:
                        continue
                    similarity = compute_cosine_similarity(
                        embeddings[name1]["embedding"], embeddings[name2]["embedding"]
                    )
                    if similarity >= batch_threshold:
                        results.append({
                            "Image 1": name1,
                            "Image 2": name2,
                            "Similarity": similarity,
                        })

            if results:
                results_df = pd.DataFrame(results).sort_values(by="Similarity", ascending=False)
                st.write("Batch Similarity Results", results_df)

                st.subheader("Visual Comparison")
                for _, row in results_df.iterrows():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(embeddings[row["Image 1"]]["image"], caption=row["Image 1"], width=200)
                    with col2:
                        st.image(embeddings[row["Image 2"]]["image"], caption=f"{row['Image 2']} (Similarity: {row['Similarity']:.2f})", width=200)
            else:
                st.warning("No matches found above the threshold.")
        else:
            st.warning("No files uploaded for batch comparison.")

# Tab 3: Manage Database
with tab3:
    st.header("Manage Database")
    uploaded_db_files = st.file_uploader("Upload Images (JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if st.button("Add to Database"):
        if uploaded_db_files:
            add_to_database(uploaded_db_files)
        else:
            st.warning("No files selected.")

    if st.button("Clear Database"):
        dbx.files_delete_v2(DROPBOX_FOLDER)
        dbx.files_create_folder(DROPBOX_FOLDER)
        image_metadata.clear()
        save_metadata(image_metadata)
        st.success("Database cleared.")

    st.write(f"Total Images in Database: {len(image_metadata)}")

# Tab 4: Algorithm Details
with tab4:
    st.header("How the App Works")
    st.subheader("Algorithm Details")
    st.markdown("""
    ### Step 1: Preprocessing
    - Uploaded images are resized to 224x224 pixels to match the input size required by the MobileNetV2 model.
    - Images are normalized using the `preprocess_input` function from TensorFlow's MobileNetV2.

    ### Step 2: Feature Extraction
    - The pre-trained **MobileNetV2** model is used to extract feature embeddings from the images.
    - These embeddings are high-dimensional vectors that represent the key features of the image.

    ### Step 3: Storing Images and Embeddings
    - Each image is assigned a unique hash key based on its content.
    - Images and their embeddings are saved in a local directory and metadata file for efficient retrieval.

    ### Step 4: Similarity Computation
    - For a new image, its embedding is compared with all stored embeddings in the database.
    - **Cosine Similarity** is used to measure how similar two embeddings are.
      - Cosine Similarity = `(A · B) / (||A|| * ||B||)`
      - A score closer to 1 indicates higher similarity.

    ### Step 5: Results Visualization
    - Images with similarity scores above the threshold are displayed in a grid layout.
    - A table lists the matched images along with their similarity scores.
    """)
