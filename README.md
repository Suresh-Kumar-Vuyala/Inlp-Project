# Parse-Force: Hierarchical Stacking Ensemble for Emotion-Cause Analysis

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

Parse-Force is an advanced NLP system implementing a hierarchical stacking ensemble for emotion-cause pair extraction in conversations, designed for SemEval-2024 Task 3. The architecture combines multiple expert models (RoBERTa, DeBERTa, LLM-Lite) with a meta-learner (XGBoost) and a causal span extractor (SpanBERT) to achieve state-of-the-art performance in identifying emotions and their textual causes.

## 🏗️ Architecture Overview

Parse-Force employs a three-level hierarchical architecture:

### Level 0: Expert Committee
- **RoBERTa-Large**: Fine-tuned for emotion classification on conversation utterances.
- **DeBERTa-V3-Base**: Weighted loss training for robust emotion detection.
- **LLM-Lite**: Phi-3-mini-4k-instruct with QLoRA adaptation for emotion-cause reasoning.

### Level 1: Meta-Learner Judge
- **XGBoost Classifier**: Fuses 21-dimensional logit vectors + 3 dialogue meta-features.
- **Confidence Gating**: Optimal threshold (τ*) for triggering Level-2 extraction.
- **Features**: Expert probabilities + speaker shifts, utterance positions, conversation lengths.

### Level 2: Causal Span Extractor
- **SpanBERT QA Model**: Extracts precise cause spans from conversation context.
- **Constrained Prediction**: Respects utterance boundaries and applies null thresholds.

## 📊 Key Features

- **Hierarchical Ensemble**: Combines strengths of transformer models and traditional ML.
- **Confidence-Based Gating**: Prevents unnecessary extractions for low-confidence predictions.
- **SemEval-2024 Compliance**: Evaluates on Strict F1 and Proportional F1 metrics.
- **Modular Design**: Separate scripts for training, inference, and evaluation.
- **GPU Acceleration**: Optimized for CUDA with memory-efficient training.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for models and data

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Suresh-Kumar-Vuyala/Inlp-Project.git
   cd Inlp-Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install core packages:
   ```bash
   pip install torch transformers datasets peft accelerate xgboost scikit-learn pandas numpy matplotlib seaborn
   ```

3. Set up Python environment (optional but recommended):
   ```bash
   python -m venv parseforce_env
   source parseforce_env/bin/activate  # On Windows: parseforce_env\Scripts\activate
   pip install -r requirements.txt
   ```

### Data Preparation

1. Place the SemEval-2024 Task 3 dataset in `NLP_Project/data/`:
   - `Subtask_1_train.json`
   - `Subtask_1_test.json` (for evaluation)

2. Preprocess data using the provided scripts or notebooks.

3. For SpanBERT, ensure tokenized datasets are in `NLP_Project/processed_data/cause_qa_tokenized/`.

## 🏃 Training

### Individual Models

Train each expert model separately:

1. **RoBERTa**:
   ```bash
   cd roberta
   python train.py --data_dir ../NLP_Project --output_dir ../NLP_Project/models/roberta/final_model
   ```

2. **DeBERTa**:
   ```bash
   cd deberta
   python train.py --data_dir ../NLP_Project --output_dir ../NLP_Project/models/deberta/final_model
   ```

3. **LLM-Lite**:
   ```bash
   cd llm_lite
   python train.py --data_dir ../NLP_Project --output_dir ../NLP_Project/models/llm_lite/final_model
   ```

4. **SpanBERT**:
   ```bash
   cd spanbert
   python train.py --data_dir ../NLP_Project --output_dir ../NLP_Project/models/spanbert/best_model
   ```

5. **XGBoost Meta-Learner**:
   ```bash
   python xgboost_meta_learner.py train --data_dir NLP_Project --output_model NLP_Project/models/meta_learner.json
   ```

### Full Pipeline Training

Use the comprehensive training script:
```bash
python parseforce_pipeline.py  # This will run the full evaluation pipeline
```

## 🔮 Inference

### Individual Model Inference

Run inference for each model to generate logits:

1. **RoBERTa**:
   ```bash
   cd roberta
   python inference.py --model_path ../NLP_Project/models/roberta/final_model --data_dir ../NLP_Project --output_csv ../NLP_Project/models/roberta_semantic/predictions_logits_validation.csv
   ```

2. **DeBERTa**:
   ```bash
   cd deberta
   python inference.py --model_path ../NLP_Project/models/deberta/final_model --data_dir ../NLP_Project --output_csv deberta_logits_validation.csv
   ```

3. **LLM-Lite**:
   ```bash
   cd llm_lite
   python inference.py --model_path ../NLP_Project/models/llm_lite/final_model --data_dir ../NLP_Project --output_csv llm_lite_logits_validation.csv
   ```

4. **XGBoost Meta-Learner**:
   ```bash
   python xgboost_meta_learner.py inference --model_path NLP_Project/models/meta_learner.json --data_dir NLP_Project --output_csv meta_predictions.csv
   ```

### Full Pipeline Inference

Run the complete Parse-Force pipeline:
```bash
python parseforce_pipeline.py  # Loads all models and evaluates on test.json
```

## 📈 Evaluation

The system evaluates on SemEval-2024 Task 3 metrics:

- **Emotion Detection Accuracy**: Overall accuracy on emotion labels.
- **Strict F1**: Exact match on emotion + exact span match on cause.
- **Proportional F1**: Token-overlap credit for partial span matches.

Run evaluation:
```bash
python parseforce_pipeline.py
```

Expected output includes:
- Per-emotion F1 scores
- Weighted average Strict and Proportional F1
- Sample correct predictions and errors
- Custom dialogue tests

## 📁 Project Structure

```
Inlp-Project/
├── NLP_Project/
│   ├── data/                          # Raw and processed datasets
│   ├── models/                        # Trained model checkpoints
│   │   ├── roberta/
│   │   ├── deberta/
│   │   ├── llm_lite/
│   │   ├── spanbert/
│   │   └── meta_learner.json
│   ├── processed_data/                # Tokenized datasets
│   └── checkpoints/                   # Training checkpoints
├── roberta/                           # RoBERTa training/inference scripts
├── deberta/                           # DeBERTa training/inference scripts
├── llm_lite/                          # LLM-Lite training/inference scripts
├── spanbert/                          # SpanBERT training/inference scripts
├── meta_learner/                      # XGBoost meta-learner scripts
├── parseforce_pipeline.py             # Full pipeline evaluation script
├── xgboost_meta_learner.py            # Standalone XGBoost script
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── *.ipynb                            # Jupyter notebooks (original implementations)
```

## 🔧 Configuration

Key hyperparameters are defined in the scripts:

- **Batch Size**: 8-16 (adjust based on GPU memory)
- **Learning Rate**: 2e-5 for transformers, 0.1 for XGBoost
- **Epochs**: 3-8 depending on model
- **Confidence Threshold (τ*)**: ~0.5 (computed during meta-learner training)

Modify these in the respective `train.py` files or config classes.

## 📊 Results

On SemEval-2024 Task 3 validation set:

- **Emotion Accuracy**: 85.2%
- **Strict F1**: 0.423
- **Proportional F1**: 0.487

Detailed results are printed during evaluation. For the latest benchmarks, run the full pipeline.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-model`
3. Commit changes: `git commit -am 'Add new model integration'`
4. Push to branch: `git push origin feature/new-model`
5. Submit a pull request

Please ensure code follows PEP 8 standards and includes docstrings.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on SemEval-2024 Task 3: Emotion-Cause Pair Extraction in Conversations
- Models built on Hugging Face Transformers
- XGBoost for meta-learning
- Special thanks to the NLP community for open-source contributions

## 📞 Contact

Suresh Kumar Vuyala - [GitHub](https://github.com/Suresh-Kumar-Vuyala)

For issues or questions, please open a GitHub issue.

---

**Note**: This implementation assumes access to the SemEval-2024 dataset. Ensure compliance with competition rules for data usage.

