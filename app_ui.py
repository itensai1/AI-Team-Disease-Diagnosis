import streamlit as st
from PIL import Image  
from streamlit_option_menu import option_menu
import requests
import zipfile
import io
from PIL import Image
import os
import shutil
import cv2
import numpy as np
import tempfile


selected = option_menu(
    menu_title=None,  # required
    options=["Home", "Model Prediction"],  # required
    icons=["house", "robot"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
            "container": {"padding": "5!important", "background-color": "#547792"},
            "icon": {"color": "white", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin":"0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#94B4C1"},
        }
)
st.markdown("""
    <style>
        .stApp {
            background-color: #213448; 
        }
    </style>
    """, unsafe_allow_html=True)

if selected == "Home":
     st.markdown("""
    <div style='background-color: #547792; padding: 10px; border-radius: 6px;'>
        <h1 style='color: white; text-align: center;'>Welcome !!</h1>
    </div>
""", unsafe_allow_html=True)
    
     st.write("&nbsp;", unsafe_allow_html=True) 
    
     st.write("" \
     "This is a simple project about our graduation project which is a web application that uses deep learning algorithm to classify and detect "
     " Retina Images. The model is trained on a dataset of retinal images and can accurately classify them into different categories. "
     "The application is built using Streamlit, a Python library for creating web applications. The model is implemented using pytorch, ")

     for _ in range(1):
         st.write("&nbsp;", unsafe_allow_html=True) 

     st.header("Our Project consists of three main parts :")
     st.write("**First:** classification of retinal images to know if it is Diabetic Retinopathy")    
     st.write("**Second:** Object detection to detect the Diabetic Retinopathy lesions")
     st.write("**Third**: segmentation to segment the lesions from the image")

     for _ in range(3):
         st.write("&nbsp;", unsafe_allow_html=True) 

     st.header("Classification Model")
     st.write("The classification model is based on EfficientNet, a state-of-the-art convolutional neural network architecture. "
        "It is designed to achieve high accuracy while being computationally efficient. The model is trained on a large dataset of retinal images and can classify them into 5 different categories.")
     st.image("images/OIP.jpg", caption="EfficientNet Architecture", use_container_width =True)
     st.write("The model is implemented using PyTorch, a popular deep learning framework. The model is trained using transfer learning, which allows it to leverage pre-trained weights from a similar task. "
        "The model is fine-tuned on the retinal image dataset to improve its performance.")
     st.header("Model Evaluation")
     st.write("We evaluate the model using CrossEntropy loss and Accuracy metrics. The model achieves an accuracy of 95% on the test set, indicating its effectiveness in classifying retinal images.")
     st.image("images/output.png", caption="Model Evalutation", use_container_width =True)
     for _ in range(3):
         st.write("&nbsp;", unsafe_allow_html=True)
     st.header("Segmentation Model")
     st.write("The segmentation model is based on the UNet architecture, a popular choice for image segmentation tasks. "
        "It is designed to capture both local and global features of the input image, allowing it to accurately segment the lesions from the background. "
        "The model is trained on a dataset of retinal images with annotated lesions, enabling it to learn the characteristics of the lesions.")
     st.image("images/OIP1.jpg", caption="UNet Architecture", use_container_width =True)
     st.write("The model is implemented using PyTorch and the segmentation_models_pytorch library, which provides a collection of state-of-the-art segmentation models. "
        "The model is trained using a combination of binary cross-entropy loss and dice loss, which helps to improve the segmentation performance.")
     st.header("Model Evaluation")
     st.write("We evaluate the model using Intersection over Union (IoU) and Dice coefficient metrics. The model achieves an IoU of 0.85 and a Dice coefficient of 0.90 on the test set, indicating its effectiveness in segmenting the lesions.")
     for _ in range(3):
         st.write("&nbsp;", unsafe_allow_html=True)

     st.header("Detection Model")
     st.write("The detection model is based on the Roboflow 3.0 Object Detection, a state-of-the-art object detection model. ")

     st.header("Model Evaluation")
     st.image("images/results.png", caption="Model Evalutation", use_container_width =True)
     st.image("images/download.png", caption="Model Evalutation", use_container_width =True)


     st.header("Future Work")
     st.write("In the future, we plan to improve the model's performance by using more advanced architectures and techniques. "
        "We also plan to expand the dataset to include more diverse retinal images, which will help to improve the model's generalization ability. "
        "Additionally, we plan to deploy the model as a web application, allowing users to easily upload and classify retinal images.")
     

     st.header("Conclusion")
     st.write("In conclusion, our web application provides a user-friendly interface for classifying and detecting diabetic retinopathy lesions in retinal images. "
        "The application leverages state-of-the-art deep learning models to achieve high accuracy and efficiency. "
        "The application can be used by healthcare professionals to assist in the diagnosis of diabetic retinopathy, ultimately improving patient outcomes.")







elif selected == "Model Prediction":
        st.title("Model Prediction")

    # Upload image
        uploaded_file = st.file_uploader("Choose an image, Please ", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

        # Prediction logic
        if st.button("Predict"):
            if "uploaded_file" in st.session_state:
                file = st.session_state.uploaded_file
                files = {
                    "file": (file.name, file.getvalue(), file.type)
                }
                response = requests.post("http://127.0.0.1:8000/predict_classification/", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.session_state.result = result  # ✅ Store result
                    st.success(f"Classification Type: {result['label']} (Confidence: {result['confidence']:.2f})")
                else:
                    st.error("Error in prediction")

        # Show stored result (if available)
        if "result" in st.session_state:
            result = st.session_state.result
            #st.info(f"Classification Type: {result['label']} (Confidence: {result['confidence']:.2f})")

            if result["label"] == "diabetic_retinopathy":
                if st.button("Segmentation"):
                    file = st.session_state.uploaded_file
                    files = {
                        "file": (file.name, file.getvalue(), file.type)
                    }
                    response = requests.post("http://127.0.0.1:8000/predict_segmentation/", files=files)

                    if response.status_code == 200:
                        try:
                            if os.path.exists("temp_masks"):
                                shutil.rmtree("temp_masks")
                            zip_bytes = io.BytesIO(response.content)
                            with zipfile.ZipFile(zip_bytes, "r") as zip_ref:
                                zip_ref.extractall("temp_masks")
                                st.success("Masks extracted and saved to 'temp_masks' folder.")
                                image_files = zip_ref.namelist()
                                cols  =  st.columns(5)

                                for i , name in enumerate(image_files):
                                    image_path = os.path.join("temp_masks", name)   
                                    image = Image.open(image_path)

                                    with cols[i%5]:
                                        st.markdown(f"**{name.replace('.png', '').replace('_', ' ').title()}**")
                                        st.image(image, use_container_width=True)
                                
                        except zipfile.BadZipFile:
                            st.error("The response content is not a valid ZIP file.")

                if st.button("Detection"):
                    file = st.session_state.uploaded_file
                    image_pil = Image.open(file).convert("RGB")
                    
                    # Create files dictionary for the request
                    files = {
                        "file": (file.name, file.getvalue(), file.type)
                    }

                    response = requests.post("http://127.0.0.1:8000/predict_detection/", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        axx = result["axx"]
                        lis = result["lis"]
                        
                        # Convert PIL image to CV2 image
                        image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                        
                        for i, (x1, y1, x2, y2) in enumerate(axx):
                            # Draw a thicker, more visible rectangle (e.g., yellow)
                            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 255), 4)  # (B, G, R), thickness=4

                            # Prepare label text
                            label = f"{lis[i]['class']} ({lis[i]['confidence']:.2f})"
                            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            # Draw filled rectangle for text background
                            cv2.rectangle(image_cv2, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 255), -1)
                            # Draw the text (black for contrast)
                            cv2.putText(image_cv2, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                        # Show result in Streamlit
                        st.image(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)
                    

            else:
                st.warning("No lesions detected, no need for segmentation.")    
            

        







   
