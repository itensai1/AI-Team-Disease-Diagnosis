import cv2
import tempfile
from PIL import Image
import app_ui as st
from roboflow import Roboflow


def detect_disease(image):
    

                    # Save temp file to read with OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        image_cv2 = cv2.imread(temp.name)

    # Get original image size
    orig_h, orig_w = image_cv2.shape[:2]

    # Run Roboflow prediction
    rf = Roboflow(api_key="cYgaBLlAQ4X9Fsmed1lb")
    project = rf.project('my-first-project-6oill')
    model = project.version(5).model
    response = model.predict(temp.name, confidence=5, overlap=50).json()

    # Roboflow resizes images to 640x640 by default
    rf_w, rf_h = 640, 640

    # Calculate scaling ratios
    x_ratio = orig_w / rf_w
    y_ratio = orig_h / rf_h

    # Draw bounding boxes
    new_pred = []
    lis = []
    for pred in response["predictions"]:
        if pred["class"] not in lis:
            lis.append(pred["class"])
            new_pred.append(pred)
    
    axx = []
    for pred in new_pred:
        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]
        
        x1 = int((x - w / 2))
        y1 = int((y - h / 2) )
        x2 = int((x + w / 2) )
        y2 = int((y + h / 2))

        axx.append((x1,y1,x2,y2))

    return axx,new_pred


        
        
        
        
        

      
   
    

