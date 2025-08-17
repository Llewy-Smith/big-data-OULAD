<!-- README updated with assistance from Claude Sonnet 4 by Anthropic -->
# Predicting Student Course Failure (OULAD)

Machine learning system for early identification of at-risk university students using interpretable AI and the Open University Learning Analytics Dataset (OULAD).

## 📊 Project Overview

Develops predictive models for early identification of student failure in university courses, completed as part of an assessment task in the **Master of Artificial Intelligence and Machine Learning** program at the University of Adelaide.

### 🎯 Research Objective

**Primary Research Question:** *Within the first 25% of teaching time, how accurately and transparently can we predict which enrolments will result in a failed course outcome?*

## 🤖 Analysis Components

### Multi-Model Architecture
- **Convolutional Neural Networks (CNN)**: 1D convolutions for tabular data analysis
- **Multi-Layer Perceptron (MLP)**: Deep neural networks with regularisation
- **Random Forest**: Ensemble baseline with feature importance analysis
- **Logistic Regression**: Interpretable linear baseline model

### Advanced Feature Engineering
- **Course Difficulty Scoring**: Replaces categorical codes with numerical failure rates
- **Early Assessment Features**: Performance indicators from first 25% of course
- **Behavioral Analytics**: VLE engagement patterns and study consistency
- **Ethical AI**: Bias detection and removal of discriminatory features

### Technical Implementation
- **Hyperparameter Optimisation**: Grid search with RFE feature selection
- **Class Balancing**: SMOTE oversampling for imbalanced datasets
- **Model Interpretability**: LIME explanations for individual predictions
- **Data Leakage Prevention**: Proper temporal splitting and feature engineering

## 📁 Repository Structure

```
├── Part_B.ipynb          # Initial baseline analysis with Random Forest
├── Part_D.ipynb          # Advanced multi-model pipeline with ethical AI
├── Part D.pdf            # Original baseline report
└── README.md             # This file
```

## 📈 Dataset Information

This project uses the **Open University Learning Analytics Dataset (OULAD)**, which contains data from courses presented by The Open University (UK) and student interactions with Virtual Learning Environment.

### Dataset Access

⚠️ **Note**: Due to GitHub size limitations, the OULAD dataset is not included in this repository.

**To get the dataset:**
1. Download from the official source: https://analyse.kmi.open.ac.uk/open-dataset
2. Extract the CSV files
3. Place them in the following directory structure:
   ```
   ../Datasets/Open University Learning Analytics Dataset (OULAD)/
   ```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- CUDA-compatible GPU (optional, for CNN training acceleration)

### Required Libraries

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow keras lime shap imblearn
```

## 🚀 Getting Started

### Advanced Pipeline (Recommended)
1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd [repo-name]
   ```

2. **Download and setup the dataset** (see Dataset Access section)

3. **Run the advanced analysis:**
   ```bash
   jupyter notebook Part_D.ipynb
   ```

### Baseline Analysis
For the original Random Forest approach:
```bash
jupyter notebook Part_B.ipynb
```

## 🔑 Key Innovations

- **Interpretable Course Features**: Numerical difficulty scores instead of cryptic codes
- **Ethical AI Implementation**: Bias detection and fair feature selection
- **Multi-Model Comparison**: CNN, MLP, Random Forest, and Logistic Regression
- **Individual Explanations**: LIME analysis for every prediction with actionable insights

## 🎓 Academic Context

- **Institution**: University of Adelaide
- **Program**: Master of Artificial Intelligence and Machine Learning
- **Focus**: Ethical AI, educational data mining, and interpretable machine learning

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome! Areas of particular interest include ethical AI improvements, novel model architectures, and enhanced interpretability methods.

## 📄 License

This project is for educational and research purposes. Please refer to the OULAD dataset license for data usage terms.

## 🙏 Acknowledgments

- **Open University** for providing the OULAD dataset and advancing educational analytics
- **University of Adelaide** for academic supervision and computational resources
- **Claude Sonnet 4 (Anthropic)** for assistance with README documentation
- The open-source community for excellent machine learning libraries

---

*For technical details, refer to Part_D.ipynb for the advanced pipeline or Part_B.ipynb for the baseline approach.*
