# Mitigating-computational-bias-using-Fairlearn

This lab notebook explores the use of the Fairlearn library to assess and mitigate computational bias in machine learning models, specifically within a healthcare context. The dataset used focuses on diabetes patients and tracks hospital readmissions. The notebook covers:

- **Introduction to Fairness in AI**: Overview of fairness considerations, particularly in sensitive applications like healthcare.
- **Dataset Exploration**: Analysis of demographic distributions (race, gender) and how underrepresented groups may impact model fairness.
- **Model Training**: Logistic regression model to predict high-risk cases for targeted care management programs.
- **Fairness Metrics and Mitigation**: Use of fairness metrics (false negative rate, selection rate) and application of Fairlearn’s `ThresholdOptimizer` to mitigate disparities across demographic groups.
- **Interpretability and Assessment**: Visualization and evaluation of fairness impacts, along with documentation practices like datasheets for datasets.

This notebook provides a practical example of fairness-aware model development and highlights essential considerations for ethical AI deployment.

---

### Requirements
- Python libraries: Fairlearn, scikit-learn, pandas, matplotlib, seaborn.
- Dataset: Publicly available clinical data on diabetes patient readmissions (1998-2008).

### Usage
Run this notebook in Google Colab or a local Jupyter environment to examine fairness metrics, train a model, and apply mitigation techniques to address bias in predictions.
