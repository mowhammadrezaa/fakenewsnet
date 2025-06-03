# Fake News Detection Model Analysis Report

## Train Set Performance

### Classification Report

```
              precision    recall  f1-score   support

        Real       0.78      0.66      0.72      3448
        Fake       0.89      0.94      0.92     10469

    accuracy                           0.87     13917
   macro avg       0.84      0.80      0.82     13917
weighted avg       0.87      0.87      0.87     13917

```

## Val Set Performance

### Classification Report

```
              precision    recall  f1-score   support

        Real       0.77      0.66      0.71      1169
        Fake       0.89      0.93      0.91      3470

    accuracy                           0.86      4639
   macro avg       0.83      0.80      0.81      4639
weighted avg       0.86      0.86      0.86      4639

```

## Test Set Performance

### Classification Report

```
              precision    recall  f1-score   support

        Real       0.78      0.66      0.71      1138
        Fake       0.89      0.94      0.92      3502

    accuracy                           0.87      4640
   macro avg       0.84      0.80      0.81      4640
weighted avg       0.87      0.87      0.87      4640

```

## Graph Structure Analysis

- Total number of nodes: 23196
- Number of edges: 231960
- Average node degree: 10.00

## Data Split Information

- Training set size: 13917
- Validation set size: 4639
- Test set size: 4640

## Feature Importance Analysis

The top 20 most important features for classification are shown in the feature_importance.png visualization.
