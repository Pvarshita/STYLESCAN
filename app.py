import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("C:\\Users\\varsh\\Downloads\\WISE_DL_PROJECT\\BG.png")

# Load the pre-trained model
model = load_model("C:\\Users\\varsh\\Downloads\\WISE_DL_PROJECT\\model.h5")  

# Load the label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("C:\\Users\\varsh\\Downloads\\WISE_DL_PROJECT\\label.npy", allow_pickle=True)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_fashion_category(img_array):
    result = model.predict(img_array)
    predicted_class = np.argmax(result)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

def main():
    custom_css = """
    <style>
        .title {
            font-family: "Monotype Corsiva", cursive;
            color: brown;
            font-size: 80px;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.markdown('<p class="title">Style Scan</p>', unsafe_allow_html=True)

    # File uploader for image
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with a specified width
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=False, width=300)

        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Perform fashion recognition when the user uploads an image
        predicted_label = predict_fashion_category(img_array)

        # Display the result in the Streamlit app
        st.markdown(f"<h2 style='color: brown; font-family: Times New Roman;'>Predicted Fashion Category: {predicted_label}</h2>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
