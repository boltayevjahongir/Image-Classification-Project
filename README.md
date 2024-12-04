
# Image Classification Project

## Description
This project demonstrates image classification using a pre-trained ResNet-18 model. The input image is processed and classified into one of the ImageNet classes.

## Steps to Run

### 1. Setup Environment
Ensure you have Python 3.7+ installed and a virtual environment created.

### 2. Install Dependencies
Install the required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Prepare Input Image
Place the image you want to classify in the project directory and update the `image_path` variable in the `image_classification.py` script.

### 4. Run the Script
Run the script to classify the input image:
```bash
python image_classification.py
```

### 5. View Results
The predicted class index for the image will be printed to the console.

## Files
- `image_classification.py`: The main script for image classification.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: This file with project documentation.
- `.gitignore`: Specifies files and directories to ignore in version control.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- Pillow
