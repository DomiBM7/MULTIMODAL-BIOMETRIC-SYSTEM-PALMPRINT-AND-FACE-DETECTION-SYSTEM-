# Multimodal Biometric Authentication System

## Author
Bukasa Muyombo

## Project Overview
A comprehensive multimodal biometric authentication system that combines facial and palmprint recognition. The system implements two distinct pipelines for biometric verification, utilizing different feature extraction and matching techniques.

## Features

### Pipeline 1
- Deep learning-based feature extraction using MobileNetV2
- Multiple fusion strategies:
  - Feature-level fusion
  - Score-level fusion
  - Match-level fusion
- Euclidean and Cosine similarity matching

### Pipeline 2
- PCA-based feature extraction
- Manhattan distance-based matching
- Weighted sum fusion
- Feature-level fusion with dimensionality reduction

## System Components

### Core Modules
1. **DataLoader**
   - Handles palmprint and facial image loading
   - Implements data preprocessing
   - Supports image resizing and normalization

2. **Feature Extractors**
   - MobileNetV2-based feature extraction (Pipeline 1)
   - PCA-based feature extraction (Pipeline 2)

3. **Matchers**
   - Multiple distance metrics support
   - Similarity score computation
   - Flexible matching strategies

4. **Fusion Modules**
   - Score-level fusion
   - Feature-level fusion
   - Decision-level fusion
   - Match-level fusion

5. **Performance Evaluation**
   - FAR/FRR calculation
   - Accuracy and recall metrics
   - Confusion matrix visualization
   - Performance plots and statistics

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Dataset Structure

dataset/
├── train/
│ ├── palm/
│ │ └── 1/
│ └── face/
│ └── 1/
└── test/
├── palm/
└── face/

### Running the System
```python
# Initialize the system
biometric_system = BiometricSystem(palmprint_folder, face_folder)

# Enroll users
biometric_system.enroll_user()

# Authenticate users
result = biometric_system.authenticate_user(palmprint_image, facial_image)

# Evaluate performance
biometric_system.evaluate_performance()
biometric_system.ConfusionMatrix()
```

## Performance Metrics
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- System Accuracy
- Recall Rate
- Confusion Matrix

## Visualization
- Authentic vs. Impostor Score Distributions
- FAR/FRR Comparison
- Accuracy and Recall Metrics
- Performance Statistics Plots
