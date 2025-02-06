# Paddy Field Insect Segmentation using Mask R-CNN

## Project Description

In modern agriculture, timely detection of pest infestations in paddy fields is essential to ensure healthy crop yields and minimize the use of harmful chemicals. This project focuses on developing an automated solution to segment insects from paddy field images using a state-of-the-art deep learning model: **Mask R-CNN**.

The project leverages the [Paddy Pests Dataset](https://www.kaggle.com/datasets/zeeniye/paddy-pests-dataset) from Kaggle, which contains images of paddy fields with and without insect infestations. The images that include pests come with detailed polygonal annotations, which allow the model to learn the precise shapes of the insects. In contrast, images without pests serve as negative samples to help the model distinguish between foreground (pests) and background.

Key aspects of this project include:
- **Dataset Preprocessing:** Parsing and converting polygonal annotations (in COCO format) and normalizing images.
- **Data Augmentation:** Applying transformations like flips, rotations, and brightness adjustments to boost model generalization.
- **Model Training:** Utilizing the Detectron2 framework to implement and train Mask R-CNN, leveraging its robust architecture for both object detection and instance segmentation.
- **Inference and Visualization:** Running inference on test images to visualize the segmentation masks overlaid on paddy field images.

This repository is structured to provide a complete pipeline—from data preparation and model training to evaluation and inference—making it a useful resource for anyone interested in applying computer vision techniques to precision agriculture.


## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/paddy-insect-segmentation.git
   cd paddy-insect-segmentation
  
2. **Create a Virtual Environment**
    venv
   ```bash
     python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install Dependencies**
   ```bash
     pip install -r requirements.txt

## Usage

# Data Preprocessing
  - Run the preprocessing script to load the dataset, resize images, and parse annotations:
    ```bash
       python src/preprocess.py

# Training the Model
  - Train the Mask R-CNN model by running the training script:
    ```bash
       python src/train_maskrcnn.py

- This script registers the dataset with Detectron2, configures the Mask R-CNN model, and starts training. Adjust hyperparameters in the script as needed.

# Running Inference
  - After training, run inference on a test image to visualize segmentation results:
     ```bash
         python src/inference.py --image data/test_images/example.jpg

- This script loads the trained model, performs inference on the specified image, and displays the image with predicted segmentation masks.

## Files Description

- src/preprocess.py:
- Contains functions to load the dataset, parse COCO annotations, and preprocess images (resizing and normalization).

- src/train_maskrcnn.py:
- Registers the dataset using Detectron2’s API, configures Mask R-CNN, and trains the model.

- src/inference.py:
- Loads a trained model to perform inference on new images. Accepts an image path as input and visualizes segmentation results with masks overlaid.

- src/utils.py:
- Includes helper functions for tasks such as image visualization.

- notebooks/EDA.ipynb:
- A Jupyter Notebook for exploratory data analysis to visualize the dataset and its annotations.

requirements.txt:
-     absl-py==2.1.0
      albucore==0.0.17
      albumentations==1.4.18
      annotated-types==0.7.0
      antlr4-python3-runtime==4.9.3
      black==24.8.0
      cachetools==5.5.1
      certifi==2025.1.31
      charset-normalizer==3.4.1
      click==8.1.8
      cloudpickle==3.1.1
      contourpy==1.1.1
      cycler==0.12.1
      detectron2 @ git+https://github.com/facebookresearch/detectron2.git@9604f5995cc628619f0e4fd913453b4d7d61db3f
      eval_type_backport==0.2.2
      filelock==3.16.1
      fonttools==4.55.8
      fsspec==2025.2.0
      fvcore==0.1.5.post20221221
      google-auth==2.38.0
      google-auth-oauthlib==1.0.0
      grpcio==1.70.0
      hydra-core==1.3.2
      idna==3.10
      imageio==2.35.1
      importlib_metadata==8.5.0
      importlib_resources==6.4.5
      iopath==0.1.9
      Jinja2==3.1.5
      kiwisolver==1.4.7
      lazy_loader==0.4
      Markdown==3.7
      MarkupSafe==2.1.5
      matplotlib==3.7.5
      mpmath==1.3.0
      mypy-extensions==1.0.0
      networkx==3.1
      numpy==1.24.4
      oauthlib==3.2.2
      omegaconf==2.3.0
      opencv-python-headless==4.11.0.86
      packaging==24.2
      pathspec==0.12.1
      pillow==10.4.0
      platformdirs==4.3.6
      portalocker==3.0.0
      protobuf==5.29.3
      pyasn1==0.6.1
      pyasn1_modules==0.4.1
      pycocotools==2.0.7
      pydantic==2.10.6
      pydantic_core==2.27.2
      pyparsing==3.1.4
      python-dateutil==2.9.0.post0
      PyWavelets==1.4.1
      PyYAML==6.0.2
      requests==2.32.3
      requests-oauthlib==2.0.0
      rsa==4.9
      scikit-image==0.21.0
      scipy==1.10.1
      six==1.17.0
      sympy==1.13.3
      tabulate==0.9.0
      tensorboard==2.14.0
      tensorboard-data-server==0.7.2
      termcolor==2.4.0
      tifffile==2023.7.10
      tomli==2.2.1
      torch==2.4.1
      torchaudio==2.4.1
      torchvision==0.19.1
      tqdm==4.67.1
      typing_extensions==4.12.2
      urllib3==2.2.3
      Werkzeug==3.0.6
      yacs==0.1.8
      zipp==3.20.2









