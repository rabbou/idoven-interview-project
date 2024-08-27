# Idoven ECG analysis Project

---

This repository contains the code and data for an ECG analysis project, focusing on distinguishing between Normal and Abnormal ECGs using machine learning models. The project is organized into various directories and notebooks that guide you through the entire process, from data exploration to model training and evaluation.

## Project Structure

### 1. `data/`
- **raw/**: Contains the raw ECG data files.
- **processed/**: Stores the preprocessed data that is ready for model training.

### 2. `models/`
- `cnn_model.keras`: The best trained Convolutional Neural Network (CNN) model.
- `resnet_model.keras`: The best trained ResNet model.

### 3. `notebooks/`
- **01-data-exploration.ipynb**: Notebook for initial data exploration and visualization.
- **02-data-preprocessing.ipynb**: Notebook detailing the preprocessing steps applied to the ECG data.
- **03-model-training.ipynb**: Notebook used to train the CNN and ResNet models.
- **04-results-analysis.ipynb**: Notebook for analyzing the results, including model evaluation metrics and visualizations.

### 4. `reports/`
- `cnn_history.pkl`: Training history of the CNN model.
- `resnet_history.pkl`: Training history of the ResNet model.
- `coutour_plot.png`: Contour plot of the hyperparameter optimization.
- `parameter_importances.png`: Visualization of feature importances or parameter impact.
- `tscnn_hyperparameter_study.pkl`: Study results on hyperparameter tuning.

### 5. `results/`
- `cnn_preds.npy`: Numpy array containing the predictions from the CNN model.
- `resnet_preds.npy`: Numpy array containing the predictions from the ResNet model.

### 6. `src/`
- **models/**: Contains model definitions and scripts for training and evaluation.
- **utils/**: Utility scripts used across the project.
  - `data_preprocessing.py`: Functions for preprocessing ECG data.
  - `ecg_visualization.py`: Functions for visualizing ECG data.
  - `example_physionet.py`: Example script for working with PhysioNet data.
  - `hyperparameter_tuning.py`: Script for hyperparameter tuning of models.

### 7. **Setup and Configuration Files**
- **Dockerfile**: Docker configuration file to set up the environment.
- **requirements.txt**: List of Python dependencies required for the project.
- **setup.py**: Script to install the `ecg_analysis` package.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: This README file.

## Getting Started

### Prerequisites
Ensure you have Docker installed on your system. Alternatively, you can manually install the required Python packages listed in `requirements.txt`.

### Installation

   ```bash
   docker build -t ecg-analysis .
   docker run -p 8888:8888 ecg-analysis
   ```
   This will set up a Jupyter Lab server with the project environment.

### Usage

1. **Data Exploration**:
   Open the `01-data-exploration.ipynb` notebook to explore the dataset and visualize ECG signals.

2. **Data Preprocessing**:
   Use `02-data-preprocessing.ipynb` to preprocess the raw ECG data and prepare it for model training.

3. **Model Training**:
   The `03-model-training.ipynb` notebook guides you through training both CNN and ResNet models on the processed data.

4. **Results Analysis**:
   Evaluate model performance and analyze results in the `04-results-analysis.ipynb` notebook.
