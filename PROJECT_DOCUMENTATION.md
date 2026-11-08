# Project Documentation
## Design and Implementation of a Convolutional Neural Network for Image Classification

**Application name:** Terrain AI  
**Project type:** Computer science / machine-learning project  
**Technologies:** Python, TensorFlow/Keras, InceptionV3, FastAPI, Next.js, React, Vercel and Cloudflare Tunnel  
**Source repository:** https://github.com/Masksm/cnn-image-classifier

---

## Abstract

This project presents the design and implementation of an image-classification system that identifies the scene represented in an image as one of six categories: buildings, forest, glacier, mountain, sea or street. The system uses a convolutional neural network (CNN), with transfer learning and fine-tuning based on the InceptionV3 architecture, to learn visual features and classify previously unseen images.

The dataset contains 17,034 labelled images divided into training, validation and test sets. Several model configurations were developed and evaluated, beginning with custom CNN architectures and progressing to transfer learning and fine-tuning. The best-performing configuration, a fully fine-tuned InceptionV3 model, achieved 91.70% accuracy on the held-out test set of 3,000 images.

The trained model is integrated into an application workflow. A Streamlit interface supports local use, while a Next.js website provides a browser-based interface. A FastAPI service loads the saved Keras model and exposes an image-prediction endpoint. The web frontend sends an uploaded image to this API and displays the highest-scoring predicted classes.

This document describes the problem, objectives, system requirements, dataset, methodology, architecture, implementation, evaluation, deployment arrangement and limitations of the project.

---

## 1. Introduction

### 1.1 Background

Digital images contain visual information that can be interpreted by computer systems. Image classification is a computer-vision task in which a model analyses an image and assigns it to one or more predefined categories. It is used in areas such as image organisation, environmental observation, security, agriculture, autonomous systems and content analysis.

Traditional image-classification approaches often depend on manually designed features. Deep learning, particularly convolutional neural networks, allows a model to learn useful visual features directly from image data. CNNs can learn patterns such as edges, textures, shapes and higher-level structures through successive convolutional and pooling operations.

This project applies CNN-based image classification to six scene categories and packages the trained model in an application that accepts image uploads and returns predictions.

### 1.2 Statement of the Problem

Manually identifying and sorting large collections of scene images can be slow and inconsistent. A computer-based system that can analyse an image and predict its scene category can make this process faster and more repeatable.

The technical challenge is to develop a model that generalises beyond the images used during training, evaluate its performance on unseen data, and make the resulting classifier accessible through a usable application.

### 1.3 Aim

The aim of this project is to design and implement a convolutional neural network-based system capable of classifying images into six scene categories.

### 1.4 Objectives

The objectives are to:

1. Obtain and organise a labelled image dataset containing six scene categories.
2. Prepare the images for use in a deep-learning workflow.
3. Design and train custom CNN models as baseline classifiers.
4. Apply transfer learning using the InceptionV3 architecture.
5. Fine-tune the pretrained model and compare the experimental results.
6. Evaluate the selected model using a held-out test dataset and classification metrics.
7. Save the trained model for reuse during inference.
8. Develop an application through which users can upload images and view classification results.
9. Expose the model through an API so that a separate web frontend can use it.

### 1.5 Scope

The system classifies an input image into one of six categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

The prediction pipeline accepts common image formats, converts the image to RGB, resizes it to 150 × 150 pixels and passes it to the trained model. The model returns the three highest-scoring classes.

The project focuses on scene classification. It is not an object detector, image-segmentation system or general-purpose image-understanding model.

---

## 2. System Requirements

### 2.1 Software Requirements

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep-learning framework | TensorFlow / Keras |
| CNN architecture | Custom CNN and InceptionV3 |
| Image processing | Pillow and TensorFlow image utilities |
| Numerical processing | NumPy |
| Local user interface | Streamlit |
| Prediction API | FastAPI and Uvicorn |
| Web frontend | Next.js, React and TypeScript |
| Source-code management | Git and GitHub |
| Large-file versioning | Git LFS |
| Frontend hosting | Vercel |
| Public API access during self-hosting | Cloudflare Tunnel |

Python 3.12 is the recommended environment for the documented setup. Backend dependency versions are specified in the backend requirements file.

### 2.2 Hardware Requirements

Model training and inference require a computer capable of running Python and TensorFlow. Training deep neural networks is more demanding than running predictions with an already-trained model.

The development environment used for local inference deployment is an HP EliteBook Folio 9480m with an Intel Core i7 processor and 8 GB RAM. The saved model is loaded by the local API process. Performance can vary with available memory, CPU load and TensorFlow configuration.

A GPU is not required for ordinary inference, although suitable GPU hardware can accelerate training.

---

## 3. Dataset Description and Preparation

### 3.1 Dataset Summary

The project dataset contains 17,034 JPG images arranged into training, validation and test subsets.

| Subset | Number of images |
|---|---:|
| Training | 11,230 |
| Validation | 2,804 |
| Test | 3,000 |
| **Total** | **17,034** |

The six categories are represented by class-named folders.

### 3.2 Dataset Directory Structure

    dataset/
    ├── train/
    │   ├── buildings/
    │   ├── forest/
    │   ├── glacier/
    │   ├── mountain/
    │   ├── sea/
    │   └── street/
    ├── validation/
    │   └── ... same six class folders
    └── test/
        └── ... same six class folders

### 3.3 Image Preprocessing

Before classification, images are prepared consistently:

1. The image is opened and its orientation is corrected using EXIF orientation information where available.
2. The image is converted to RGB colour mode.
3. The image is resized to 150 × 150 pixels.
4. The image is converted into a numerical array.
5. A batch dimension is added before the array is passed to the model.

The training data loader uses a batch size of 32. The documented training augmentation includes random horizontal flips, small rotations and zoom operations.

The inference pipeline uses the same six-class ordering configured in the prediction module:

1. buildings
2. forest
3. glacier
4. mountain
5. sea
6. street

Maintaining the correct class order is essential because the model output positions correspond to these labels.

---

## 4. Methodology and Model Development

### 4.1 Convolutional Neural Networks

A convolutional neural network is a deep-learning architecture designed to process grid-like data such as images. It learns visual features through trainable filters.

A typical CNN contains:

- **Convolution layers:** Apply filters to identify local visual patterns.
- **Activation functions:** Introduce non-linearity so the network can learn complex relationships.
- **Pooling operations:** Reduce spatial dimensions and retain useful information.
- **Feature extraction layers:** Build increasingly abstract representations.
- **Classification layers:** Map learned features to class scores.

During training, model parameters are adjusted to reduce prediction error on the training data.

### 4.2 Custom CNN Experiments

Custom CNN models were developed first to establish baseline performance. An improved custom CNN configuration achieved 77.40% test accuracy.

These experiments provided a comparison point for evaluating transfer learning.

### 4.3 Transfer Learning with InceptionV3

InceptionV3 is a deep convolutional neural network architecture originally trained on a large image dataset. Transfer learning reuses its learned visual representations for a different classification task.

The project adapted the classification output for the six target classes. In the initial transfer-learning experiment, the pretrained feature-extraction backbone was frozen while the new classification layers were trained.

This configuration achieved 89.93% test accuracy.

### 4.4 Fine-Tuning

Fine-tuning allows selected pretrained layers to be updated using the target dataset. It can help the model adapt its learned visual representations to the new classification task.

Two fine-tuning experiments were evaluated:

- Partial fine-tuning: achieved 91.07% test accuracy.
- Full fine-tuning: achieved 91.70% test accuracy.

In the full fine-tuning configuration, Batch Normalization layers were kept frozen. The selected model is saved at:

    model/best_inception_full_finetuned.keras

The model contains approximately 22.28 million trainable parameters and 51,648 non-trainable parameters, according to the recorded project configuration.

### 4.5 Model Comparison

| Model configuration | Test accuracy |
|---|---:|
| Improved custom CNN V2 | 77.40% |
| InceptionV3 transfer learning | 89.93% |
| InceptionV3 partial fine-tuning | 91.07% |
| **InceptionV3 full fine-tuning** | **91.70%** |

The fully fine-tuned InceptionV3 model was selected because it achieved the highest recorded test accuracy among the compared configurations.

---

## 5. System Design and Architecture

### 5.1 Main Components

The implemented system has four main components:

1. **Trained model:** The saved Keras model performs image classification.
2. **Prediction module:** The shared prediction script loads the model, prepares images and returns the highest-scoring classes.
3. **Backend API:** The FastAPI application provides HTTP endpoints.
4. **User interfaces:** The Streamlit application supports local use, while the Next.js frontend provides a browser-based interface.

### 5.2 Web Deployment Architecture

The browser-based version separates the frontend from the machine-learning inference service.

    User's browser
          |
          v
    Next.js frontend on Vercel
          |
          | HTTPS request
          v
    Cloudflare Tunnel
          |
          v
    User's Windows laptop
       FastAPI / Uvicorn
          |
          v
       Prediction module
          |
          v
    InceptionV3 trained model
          |
          v
    Top-three class predictions
          |
          v
    Frontend displays the result

The frontend and model API are separate because the TensorFlow runtime and trained model are comparatively large for a typical frontend serverless deployment.

### 5.3 Prediction Workflow

The prediction process follows these steps:

1. The user selects an image in the frontend.
2. The frontend checks that the file is an image and does not exceed 10 MB.
3. The frontend sends the image to the API as multipart form data using the field name **file**.
4. FastAPI checks the uploaded content type and file size.
5. The prediction module opens and preprocesses the image.
6. The trained Keras model calculates class scores.
7. The prediction module selects the three highest-scoring classes.
8. The API returns the predictions as JSON.
9. The frontend displays the leading class and the three prediction scores.

---

## 6. Implementation Details

### 6.1 Prediction Module

The prediction module contains the reusable inference logic.

Its responsibilities include:

- Resolving the project root and model file path.
- Loading the trained model once and reusing it through an in-process cache.
- Checking that the model output contains six classes.
- Opening image bytes, file-like objects or filesystem paths.
- Correcting image orientation and converting images to RGB.
- Resizing images to 150 × 150 pixels.
- Running model inference.
- Returning the three highest-scoring predictions.

The model is loaded from the selected full-fine-tuning checkpoint.

### 6.2 FastAPI Backend

The API is implemented in the backend main module.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Returns the service status |
| GET | /health | Reports API health and whether the model file exists |
| POST | /predict | Accepts an image and returns predictions |

The prediction endpoint expects a multipart upload with a field named **file**.

The backend accepts JPEG, PNG, WEBP and BMP content types and enforces a maximum upload size of 10 MB. It returns appropriate HTTP errors for unsupported content types, empty files, oversized files and prediction failures.

The backend supports the ALLOWED_ORIGINS environment variable for configuring browser CORS origins. For the deployed website, this should be set to the exact frontend origin.

### 6.3 Prediction Response Format

A successful prediction response has this structure:

    {
      "predictions": [
        {
          "class": "buildings",
          "confidence": 0.9999
        },
        {
          "class": "sea",
          "confidence": 0.00003
        },
        {
          "class": "street",
          "confidence": 0.00002
        }
      ]
    }

The values above illustrate the response structure. Actual classes and scores depend on the uploaded image.

### 6.4 Next.js Frontend

The frontend is located in the web directory and is built with Next.js, React and TypeScript.

Its main functions are:

- Selecting an image from the user's device.
- Dragging and dropping an image into the upload area.
- Displaying an image preview.
- Checking file type and size in the browser.
- Sending the image to the configured API.
- Displaying the most likely class and the top-three scores.
- Showing errors when the API cannot be reached or returns an unsuccessful response.
- Allowing the user to reset the interface and classify another image.

The frontend reads the API base URL from the environment variable:

    NEXT_PUBLIC_API_URL

The value must be the API's base URL without a trailing slash. The frontend appends /predict when making a prediction request.

### 6.5 Streamlit Interface

The repository also contains a Streamlit application for running the classifier locally. It provides a local interface for uploading images and viewing classification output without requiring the Next.js frontend.

---

## 7. Installation and Execution

### 7.1 Clone the Repository

Git LFS is required to retrieve large model and dataset files.

    git lfs install
    git clone https://github.com/Masksm/cnn-image-classifier.git
    cd cnn-image-classifier
    git lfs pull

### 7.2 Create a Python Environment

Windows PowerShell:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1

macOS or Linux:

    python3.12 -m venv .venv
    source .venv/bin/activate

### 7.3 Install the Main Project Dependencies

    python -m pip install --upgrade pip
    pip install -r requirements.txt

### 7.4 Run the Streamlit Application

Run from the repository root:

    streamlit run app/app.py

### 7.5 Install and Run the FastAPI Backend

Install the API dependencies:

    pip install -r backend/requirements.txt

For local development, start the API from the repository root:

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Check the service at:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

The interactive API documentation is provided by FastAPI.

### 7.6 Run the Next.js Frontend

Open a second terminal and move into the frontend directory:

    cd web
    npm install
    npm run dev

Configure NEXT_PUBLIC_API_URL in the frontend environment before testing predictions. For local API testing, the value can be:

    NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

Restart the Next.js development server after changing this variable.

---

## 8. Deployment Configuration

### 8.1 Frontend Hosting

The Next.js frontend is deployed on Vercel:

- Project URL: https://terrain-ai-masksms-projects.vercel.app
- GitHub repository: https://github.com/Masksm/cnn-image-classifier
- Frontend root directory in the Vercel project: web

The Vercel project must have the environment variable NEXT_PUBLIC_API_URL configured with the public API base URL. A new deployment is required after changing this build-time variable.

### 8.2 Self-Hosted API and Cloudflare Tunnel

The current self-hosting approach runs the FastAPI service on the development laptop and uses Cloudflare Tunnel to provide a public HTTPS route to the local API.

Start the backend:

    $env:ALLOWED_ORIGINS="https://terrain-ai-masksms-projects.vercel.app"
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

In a separate PowerShell window, start a temporary Cloudflare Quick Tunnel:

    cloudflared tunnel --url http://127.0.0.1:8000 --edge-ip-version 4

Cloudflare prints a public URL ending in trycloudflare.com. Use that URL as the value of NEXT_PUBLIC_API_URL in Vercel, without a trailing slash, and redeploy the frontend.

**Operational requirements and limitations:**

- The laptop must remain powered on and connected to the internet.
- The FastAPI process and Cloudflare Tunnel process must both remain running.
- Closing either terminal or shutting down the laptop interrupts API access.
- A Quick Tunnel URL is temporary and can change when a new tunnel is created. Update the Vercel environment variable and redeploy if the URL changes.
- Quick Tunnels are intended for experimentation and testing, not guaranteed production uptime.
- The API has no authentication or rate-limiting mechanism in the documented implementation. CORS restricts browser access from other origins but is not authentication and does not prevent direct HTTP requests. Do not treat the public endpoint as production-secure.
- A persistent deployment should use a named Cloudflare Tunnel or a suitable managed inference host, together with API authentication, rate limiting, monitoring and a reliable always-on server.

The local API and model inference were tested successfully. The public Quick Tunnel and end-to-end Vercel-to-laptop workflow should be considered unverified until a public tunnel URL is active and a prediction has been completed through the deployed website.

---

## 9. Testing and Evaluation

### 9.1 Model Evaluation

The selected model was evaluated using a held-out test set containing 3,000 images.

The recorded overall results in the repository are:

| Metric | Result |
|---|---:|
| Accuracy | 91.70% |
| Macro precision | 91.88% |
| Macro recall | 91.94% |
| Macro F1-score | 91.87% |

### 9.2 Per-Class Performance

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| Buildings | 91.59% | 92.22% | 91.90% | 437 |
| Forest | 99.36% | 98.31% | 98.83% | 474 |
| Glacier | 88.76% | 82.82% | 85.69% | 553 |
| Mountain | 85.03% | 87.62% | 86.30% | 525 |
| Sea | 92.41% | 97.84% | 95.05% | 510 |
| Street | 94.13% | 92.81% | 93.47% | 501 |

### 9.3 Interpretation of Results

The model achieved its strongest class-level performance on forest images, with an F1-score of 98.83%. Sea and street images also achieved strong results.

Glacier and mountain scenes were more difficult to distinguish. The evaluation showed confusion between these two visually related categories. This is an important limitation because some images can contain similar textures, lighting conditions, snow, ice and mountainous terrain.

The overall test accuracy of 91.70% indicates that the selected model classified most images in the held-out test set correctly. It does not mean that every prediction is correct or that the model will achieve the same accuracy on images from every source.

### 9.4 Application and API Checks

The following local checks were completed during development:

- The FastAPI application started successfully with Uvicorn.
- The root endpoint returned the service status.
- The health endpoint reported that the model file was present.
- A test image was submitted to /predict.
- The API returned three predictions, with the expected class and confidence fields.
- The tested buildings image was classified as buildings with a high model score.

The public tunnel and deployed frontend-to-API connection require a separate end-to-end test after the tunnel is successfully established.

---

## 10. Limitations

The project has the following limitations:

1. **Limited class coverage:** The model recognises only the six categories used during training.
2. **Input resolution:** Images are resized to 150 × 150 pixels. Fine details may be lost during resizing.
3. **Similar visual categories:** Glacier and mountain scenes can share visual characteristics and are more likely to be confused.
4. **Dataset dependence:** Performance depends on the quality, diversity and distribution of the dataset. Images that differ substantially from the training data may produce incorrect predictions.
5. **Confidence interpretation:** The returned confidence values are model scores. They are not a guarantee that a prediction is correct and are not, by themselves, a reliable measure of whether an image belongs to an unknown category.
6. **Self-hosted availability:** When the API runs on a personal laptop, service availability depends on that laptop, its power supply and its internet connection.
7. **Public API security:** The current API implementation does not include authentication or rate limiting. A public tunnel should therefore be treated as a testing arrangement, not a hardened production service.
8. **Hardware constraints:** Training and serving a large TensorFlow model can consume significant memory and processing resources.

---

## 11. Recommendations for Future Work

Future improvements may include:

- Increasing the diversity and size of the training dataset.
- Investigating higher input resolutions and their effect on accuracy and resource usage.
- Improving the distinction between glacier and mountain scenes.
- Applying systematic hyperparameter tuning and additional model comparisons.
- Evaluating calibration and adding a better-supported unknown-image rejection method.
- Adding automated tests for preprocessing, API validation and prediction responses.
- Adding authentication, rate limiting, logging and monitoring to the API.
- Moving inference to a reliable always-on server or managed model-hosting service.
- Improving accessibility, mobile responsiveness and user feedback in the web interface.
- Recording model versions and experiment configurations to improve reproducibility.

---

## 12. Conclusion

This project designed and implemented a CNN-based image-classification system for six scene categories. The work progressed from custom CNN experiments to transfer learning and fine-tuning with InceptionV3. The fully fine-tuned model achieved the best recorded result, with 91.70% accuracy on a held-out test set of 3,000 images.

The trained model was integrated into a reusable prediction module and made accessible through both a local Streamlit interface and a FastAPI service. A Next.js frontend provides a separate browser-based interface that communicates with the prediction API.

The project demonstrates the application of deep learning to image classification, the evaluation of alternative model configurations, and the integration of a trained model into a usable software system. Although the results are strong on the selected test set, further work is required to improve generalisation, reduce class confusion and provide secure, reliable public hosting.

---

## Appendix A: Important Project Files

| Path | Description |
|---|---|
| app/app.py | Streamlit application |
| src/predict.py | Shared image preprocessing and inference logic |
| src/ | Model development, training and evaluation scripts |
| model/best_inception_full_finetuned.keras | Selected trained model |
| backend/main.py | FastAPI application |
| backend/requirements.txt | Backend Python dependencies |
| backend/Dockerfile | Container build configuration |
| web/app/page.tsx | Main Next.js user interface |
| web/package.json | Frontend dependencies and scripts |
| results/ | Evaluation reports, plots and experiment outputs |
| dataset/ | Training, validation and test images |

## Appendix B: Useful Commands

Activate the Windows environment:

    .\.venv\Scripts\Activate.ps1

Start the local prediction API:

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Check API health:

    Invoke-RestMethod http://127.0.0.1:8000/health

Test a prediction from the repository root:

    curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@dataset/test/buildings/20057.jpg"

Start the local Streamlit interface:

    streamlit run app/app.py

Start the frontend:

    cd web
    npm run dev

---

**Documentation note:** This document describes the implementation and results recorded in the repository at the time of writing. Any deployment status explicitly marked as unverified must be tested before being presented as a completed live deployment.
