# CNN Image Classification System

> **Full project documentation:** See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the project background, objectives, methodology, architecture, implementation, testing, results, deployment and limitations.

A TensorFlow/Keras image-classification project that predicts one of six scene categories: **buildings, forest, glacier, mountain, sea, and street**. The project compares custom CNN models with transfer learning and fine-tuning using InceptionV3, then exposes the best model through a Streamlit application.

## Results at a glance

The best-performing model is the fully fine-tuned InceptionV3 model.

| Model | Test accuracy |
|---|---:|
| Custom CNN V2 | 77.40% |
| InceptionV3 transfer learning (frozen backbone) | 89.93% |
| InceptionV3 partial fine-tuning | 91.07% |
| **InceptionV3 full fine-tuning** | **91.70%** |

The final result was measured on the held-out test set of **3,000 images**. The classification report records 91.70% accuracy, 91.88% macro precision, 91.94% macro recall, and 91.87% macro F1-score.

### Per-class test performance

| Class | Precision | Recall | F1-score | Test images |
|---|---:|---:|---:|---:|
| Buildings | 91.59% | 92.22% | 91.90% | 437 |
| Forest | 99.36% | 98.31% | 98.83% | 474 |
| Glacier | 88.76% | 82.82% | 85.69% | 553 |
| Mountain | 85.03% | 87.62% | 86.30% | 525 |
| Sea | 92.41% | 97.84% | 95.05% | 510 |
| Street | 94.13% | 92.81% | 93.47% | 501 |

The most noticeable remaining weakness is confusion between **glacier** and **mountain** scenes. See the evaluation plots and reports in `results/`.

## Dataset

The repository contains 17,034 JPG images arranged in class subfolders:

- Training: 11,230 images
- Validation: 2,804 images
- Test: 3,000 images

Expected layout:

```text
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
```

Images are resized to **150 × 150** pixels by the data loader. The dataset loader uses integer labels and a batch size of 32. Training augmentation includes random horizontal flips, small rotations, and zoom.

## Model development

The experiments progress from custom convolutional neural networks to transfer learning:

1. Baseline custom CNN.
2. Improved custom CNN and a second version.
3. InceptionV3 transfer learning with the pretrained feature extractor frozen.
4. Partial fine-tuning of InceptionV3.
5. Full fine-tuning of InceptionV3, with Batch Normalization layers kept frozen.

The selected inference model is:

`model/best_inception_full_finetuned.keras`

Other best and final checkpoints are retained in `model/` so the experiments remain available for comparison.

## Repository structure

```text
.
├── app/
│   └── app.py                 # Streamlit user interface
├── dataset/                   # Train, validation and test images
├── model/                     # Saved Keras model checkpoints
├── results/                   # Reports, plots, predictions and training logs
└── src/                       # Data loading, model building, training and evaluation
```

Large `.keras` model files and `.jpg` dataset images are stored with **Git LFS**.

## Run the application locally

### 1. Prerequisites

- Python 3.12 recommended
- Git
- Git LFS

Install Git LFS and initialize it once:

```bash
git lfs install
```

### 2. Clone the repository

```bash
git clone https://github.com/Masksm/cnn-image-classifier.git
cd cnn-image-classifier
git lfs pull
```

### 3. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Start the application

Run this from the repository root:

```bash
streamlit run app/app.py
```

The application opens in your browser. Upload a supported image and select **Classify Image**. The interface displays the predicted category, confidence score, and up to three highest-scoring classes.

## Run training and evaluation scripts

Run commands from the repository root so relative dataset and output paths resolve correctly. For example:

```bash
python src/train_full_finetune.py
python src/evaluate_full_finetuned.py
```

Training InceptionV3 from scratch or fine-tuning it can require substantial memory, compute, and time. The saved best model is already included, so training is not required to use the Streamlit application.

## Important notes and limitations

- A 91.70% test accuracy is the result on this dataset and split; it does not guarantee the same performance on unrelated real-world images.
- The application uses a 49% confidence threshold as a simple rejection heuristic. It is not a calibrated out-of-distribution detector, and confidence should not be interpreted as a guarantee of correctness.
- Glacier and mountain images are the most challenging classes in the current evaluation.
- The model expects the same six class ordering used during training: buildings, forest, glacier, mountain, sea, street.

## Results and reproducibility

Evaluation reports, confusion matrices, training logs, and comparison plots are stored in `results/`. The saved models and image dataset are tracked through Git LFS; install Git LFS before cloning so the actual large files can be downloaded.

## License

No license has been added yet. Unless a license is specified, the repository's contents should not be assumed to be available for unrestricted reuse.
