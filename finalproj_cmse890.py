import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import joblib

# Set page config
st.set_page_config(page_title='Bacterial Concentration Estimation via Image Classification', page_icon=None, layout='wide')

# Set background color and padding
st.markdown(
    """
    <style>
    body {
        background-color: #001F3F;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load saved ensemble model
model_path = 'eclf.pkl'
model = joblib.load(model_path)

# Define function for preprocessing image
def preprocess_image(image):
    try:
        # Open image and rotate if necessary
        img = Image.open(image)
        if hasattr(img, '_getexif'):  # only for JPEGs
            exifdata = img._getexif()
            if exifdata is not None:
                orientation = exifdata.get(274, None)
                if orientation is not None:
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)

        # Crop and resize image
        width, height = img.size
        if width > height:
            left = int((width - height) / 2 + (height - 800) / 2)
            upper = int((height - 1350) / 2)
            right = int((width + height) / 2 - (height - 800) / 2)
            lower = int((height + 1200) / 2)
        else:
            left = int((width - 800) / 2)
            upper = int((height - width) / 2 + (width - 1350) / 2)
            right = int((width + 800) / 2)
            lower = int((height + width) / 2 - (width - 1200) / 2)
        cropped_image = img.crop((left, upper, right, lower))
        resized_image = cropped_image.resize((150, 150)).convert('L')

        # Convert images to arrays
        processed_image = np.array(resized_image).reshape(1, -1)
        processing_steps = ""
        return processed_image, processing_steps

    except:
        st.write('Failed to preprocess image.')
        return None, None


# Define Streamlit app
st.title('Bacterial Concentration Estimation via Image Classification')
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg'])

# Add sidebar with abstract
st.sidebar.title('Abstract')
st.sidebar.write('by Angelica Louise M. Gacis')
st.sidebar.write('M.S. in Data Science, Michigan State University')
st.sidebar.write('[Link to repository](https://github.com/yourusername/yourrepository)')
st.sidebar.markdown('Recent advances in machine learning have shown great potential in providing more efficient and reliable methods for bacterial enumeration. This study aimed to develop a machine learning model that can accurately detect the presence or absence of bacterial cells in test tube samples of carbohydrate-coated magnetic nanoparticles. The input data for the model consisted of 120 individual photographs of test tubes, each containing a different sample. The output of the model is a binary classification of the samples, indicating whether bacterial cells are present. Two main approaches were utilized in this study: applying various binary, multi- class, and regression models from the Scikit Learn library to the image data and defining a Convolutional Neural Network (CNN) using the Keras Sequential API and training it on the image data. The results showed that the ensemble model performed better, achieving an accuracy of 0.89 compared to the CNN’s accuracy of 0.7. Further recommendations for improving the study include utilizing augmented data for the Scikit Learn models, increasing the dataset size, and exploring other deep learning architectures such as recurrent neural networks (RNNs) and long short-term memory (LSTM) networks using the TensorFlow library. Overall, the development of a machine learning algorithm for bacterial enumeration has the potential to provide faster and more consistent methods for detecting bacterial contamination in various industries, ultimately contributing to improved public health and safety.')
# st.sidebar.title('Angelica Louise M. Gacis')
# st.sidebar.write('M.S. in Data Science, Michigan State University')

if uploaded_file is not None:
    # Preprocess uploaded image
    input_image, processing_steps = preprocess_image(uploaded_file)
    
    if input_image is not None:
        # Make prediction with model
        prediction = model.predict_proba(input_image.reshape(1, -1))
        predicted_class = prediction[:,1] > 0.5
        
        # Write result
        if predicted_class:
            st.write("<p style='font-size: 24px; text-align:center;'>Prediction: Bacterial Cells Present</p>", unsafe_allow_html=True)
        else:
            st.write("<p style='font-size: 24px; text-align:center;'>Prediction: Bacterial Cells Absent</p>", unsafe_allow_html=True)

        st.write("<p style='font-size: 16px; text-align:center;'>Class probabilities:</p>", unsafe_allow_html=True)
        st.write(f"<p style='font-size: 16px; text-align:center;'>{prediction}</p>", unsafe_allow_html=True)

            
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded image', use_column_width=True)
        
        # Display processed image and processing steps
        st.write(processing_steps)
        st.image(Image.fromarray(input_image.reshape(150,150)), caption='Processed image', use_column_width=True)

