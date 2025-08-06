import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load your precomputed image embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Create a Streamlit app
st.title('Fashion Recommender System')

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        return str(e)

# Function for feature extraction
def feature_extraction(img_path, model):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Extract features using the model
    result = model.predict(preprocessed_img).flatten()
    
    # Normalize the feature vector
    normalized_result = result / norm(result)
    
    return normalized_result


# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# Upload an image
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    upload_result = save_uploaded_file(uploaded_file)
    if upload_result == 1:
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)

        # Recommendation
        indices = recommend(features, feature_list)

        # Display similar images
        st.header('Similar Images:')
        col1, col2, col3, col4, col5 = st.columns(5)
        for i in indices[0]:
            st.image(filenames[i], use_column_width=True)

    else:
        st.error(f"Error during file upload: {upload_result}")

# Create an 'uploads' directory to store uploaded images
os.makedirs('uploads', exist_ok=True)
