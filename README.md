# Emotion-Cause Detection using Transformer Models

This project implements a multi-model pipeline for Emotion and Cause Detection using transformer-based architectures. The system combines DeBERTa, RoBERTa, and SpanBERT models to perform emotion classification and cause extraction from text.

The final predictions are generated through an integration pipeline that merges outputs from different models.

---

# Project Structure

project/

├── DebertaModel.ipynb  
├── RobertaModel.ipynb  
├── SpanbertaModel.ipynb  
├── Intergration.ipynb  
└── README.md  

---

# Overview

The project follows a multi-stage NLP pipeline.

1. Emotion Detection  
Predicts the emotion expressed in the text.

Models used:
- DeBERTa
- RoBERTa

2. Cause Extraction  
Identifies the span of text responsible for the detected emotion.

Model used:
- SpanBERT

The task is formulated as a Question Answering problem where the model predicts the start and end positions of the cause span.

3. Integration  
Outputs from different models are combined to produce the final structured result containing:

- Input text
- Predicted emotion
- Extracted cause

---

# Models

DeBERTa

Notebook: DebertaModel.ipynb

Purpose:  
Multi-label emotion classification using HuggingFace Transformers and PyTorch.

Evaluation Metrics:
- Precision
- Recall
- F1 Score

---

RoBERTa

Notebook: RobertaModel.ipynb

Purpose:  
Transformer-based model for emotion classification using HuggingFace libraries.

---

SpanBERT

Notebook: SpanbertaModel.ipynb

Purpose:  
Extract the cause of an emotion from the text.

Approach:  
The task is converted into a Question Answering problem where the model predicts the text span that answers the cause question.

---

# Integration Pipeline

Notebook: Intergration.ipynb

Steps:

1. Load emotion predictions  
2. Run cause extraction  
3. Combine outputs  
4. Produce final structured results  

Example Output:

{
"text": "I was upset because my project got rejected.",
"emotion": "sadness",
"cause": "my project got rejected"
}

---

# Trained Models

The trained models are stored in Google Drive due to their large size.

Google Drive link below:

[CLICK_HERE](https://drive.google.com/drive/folders/12_TspcuD5PDrakWd_a7zAMItwS2IXYtN?usp=sharing)

---

# Installation

Install required libraries:

pip install transformers  
pip install datasets  
pip install torch  
pip install scikit-learn  
pip install accelerate  
pip install pandas  
pip install numpy  

---

# Running the Project

1. Run DebertaModel.ipynb  
2. Run RobertaModel.ipynb  
3. Run SpanbertaModel.ipynb  
4. Run Intergration.ipynb  

---

# Hardware Requirements

Recommended environment:

- GPU with CUDA support
- Google Colab environment

---

# Libraries Used

- HuggingFace Transformers  
- HuggingFace Datasets  
- PyTorch  
- Scikit-learn  
- NumPy  
- Pandas  

---

# Future Improvements

- Better ensemble strategies  
- Improved span extraction  
- Hyperparameter tuning  
- Larger datasets  

---

