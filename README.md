# Retinal Disease Detection and Classification System

## Overview
This project is a comprehensive web application for analyzing retinal images to detect and classify various eye conditions, with a particular focus on Diabetic Retinopathy. The system combines three powerful deep learning models for classification, segmentation, and lesion detection.

## Features
- **Image Classification**: Identifies whether an image shows signs of Diabetic Retinopathy
- **Lesion Segmentation**: Segments different types of lesions in retinal images
- **Object Detection**: Detects and localizes specific lesions and anomalies
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Real-time Processing**: Provides immediate results for uploaded images

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Deep Learning**: PyTorch, EfficientNet, UNet
- **Object Detection**: Roboflow
- **Image Processing**: OpenCV, PIL
- **Dependencies Management**: Pipenv

## Models
1. **Classification Model**
   - Architecture: EfficientNet-B0
   - Purpose: Binary classification of retinal images
   - Performance: 95% accuracy on test set

2. **Segmentation Model**
   - Architecture: UNet with ResNet34 backbone
   - Purpose: Segmentation of five different types of lesions
   - Features detected:
     - Optic Disc
     - Microaneurysms
     - Hemorrhages
     - Soft Exudates
     - Hard Exudates

3. **Detection Model**
   - Platform: Roboflow
   - Purpose: Object detection and localization of lesions
   - Features: Real-time detection with bounding boxes

## Installation

### Prerequisites
- Python 3.10
- Pipenv

### Setup
1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pipenv install
```

3. Configure the application:
   - Update `config.json` with your settings
   - Ensure model weights are in the correct location

### Running the Application

1. Start the FastAPI backend server:
```bash
./bin/start_server.bat
```

2. Start the Streamlit frontend:
```bash
./bin/start_streamlit.bat
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

## Project Structure
```
├── app_ui.py                # Streamlit frontend
├── disease_dignoses/
│   ├── api.py              # FastAPI backend
│   ├── classifier/         # Classification model
│   ├── segmentation/      # Segmentation model
│   └── detection/         # Detection model
├── bin/                    # Startup scripts
├── assets/                # Model weights
└── config.json            # Configuration file
```

## Configuration
The `config.json` file contains important settings:
```json
{
    "PRE_TRAINED_MODEL_CLASSIFIER": "assets/efficientnet_model.pth",
    "PRE_TRAINED_MODEL_SEGMENTOR": "assets/unet_80.pth",
    "num_classes": 4,
    "IMG_SIZE": 640,
    "NUM_CLASSES": 5
}
```

## API Endpoints

### Classification
- **Endpoint**: `/predict_classification/`
- **Method**: POST
- **Input**: Image file
- **Output**: Classification result with confidence score

### Segmentation
- **Endpoint**: `/predict_segmentation/`
- **Method**: POST
- **Input**: Image file
- **Output**: ZIP file containing segmentation masks

### Detection
- **Endpoint**: `/predict_detection/`
- **Method**: POST
- **Input**: Image file
- **Output**: Bounding boxes and lesion classifications

## Future Work
- Model performance improvements
- Dataset expansion
- Enhanced UI features
- Mobile application development
- Integration with medical record systems

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Authors
* Shawky Gamal [Git Hub](https://github.com/shawky-gamal-22) [LinkedIn](https://www.linkedin.com/in/shawky-gamal-0712b220a/)
* Kareem Ashraf [Git Hub](https://github.com/karim3421) [LinkedIn](https://www.linkedin.com/in/karim-ashraf-80a867229/)

## Acknowledgments
- EfficientNet architecture
- UNet architecture
- Roboflow platform
- PyTorch community
- Streamlit team
