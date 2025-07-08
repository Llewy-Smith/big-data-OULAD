# Predicting Student Course Failure (OULAD)

This project explores early prediction of student failure in university courses using the Open University Learning Analytics Dataset (OULAD). It was completed as part of an assessment task in the Master of Artificial Intelligence and Machine Learning program at the University of Adelaide.

## Research Objective

**Primary question:**  
Within the first 25% of teaching time, how accurately and transparently can we predict which enrolments will result in a failed course outcome?

The analysis includes:
- Feature extraction across engagement, demographic, administrative, and course metadata
- Exploratory pattern analysis
- Early baseline modelling using Random Forest
- Justification for feature engineering and class balancing strategies (e.g., SMOTE)
- Visualisations for feature interaction and class imbalance

## Files in this Repository

- `Part_B.ipynb`: Jupyter Notebook with full code and outputs  
- `Part B.pdf`: Final submitted report  

## Dataset Access

Due to size limitations, the OULAD dataset is not included in this repository. You can download the dataset directly from the Open University at:  
[https://analyse.kmi.open.ac.uk/open-dataset](https://analyse.kmi.open.ac.uk/open-dataset)

After downloading, place the `.csv` files into the following directory relative to the notebook: `../Datasets/Open University Learning Analytics Dataset (OULAD)/`.

## Required Libraries

This project uses the following Python libraries:

```python
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
```
