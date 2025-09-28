# Paw-Classifier
This project implements an efficient deep learning solution for binary image classification — distinguishing between cats and dogs.
📚 Project Description
This project implements a deep learning model to classify images of cats and dogs using MobileNetV2 with transfer learning. The focus is on achieving high accuracy while ensuring the model remains lightweight and deployable on resource-constrained devices.

The model is trained using a balanced dataset achieved through SMOTE, preventing bias toward one class. Real-time predictions are made possible through an OpenCV-based pipeline.

📊 Dataset Used
Name: Kaggle Dogs vs. Cats Dataset
Description: 25,000 labeled images of cats and dogs.

⚙️ Technologies and Libraries
Python 3.8+
TensorFlow/Keras: For model development and training.
OpenCV: For image processing and real-time prediction.
Scikit-learn: For SMOTE and evaluation metrics.
Matplotlib/Seaborn: For data visualization.

📈 Model Architecture
Base Model: MobileNetV2 (pre-trained on ImageNet).
Classifier Head: Custom dense layers for binary classification (cat vs dog).
Optimization: Adam optimizer with Categorical Crossentropy loss.

Training Split:
80% Training Data
20% Testing Data

🏆 Model Performance
Metric	Value
Accuracy	91.2%
Precision	91.8%
Recall	90.9%
F1 Score	91.35%

🔥 How to Run
Clone this repository:
git clone https://github.com/yourusername/cat-vs-dog-classification.git
cd cat-vs-dog-classification

OR
download the file and dataset then run on Google Collab.

Install dependencies:
pip install -r requirements.txt

Train the model or load the pre-trained model:
python train_model.py

Run real-time prediction:
python predict_realtime.py

📑 References
MobileNetV2 Paper
Kaggle Dogs vs Cats Dataset
TensorFlow and Keras Official Documentation.
