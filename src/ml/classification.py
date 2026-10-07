"""
ml/classification.py
Modular, leakage-free classification pipeline for resume job-category prediction.

Key ML Decisions:
1. sklearn Pipeline: Chains TfidfVectorizer -> LogisticRegression so that
   vocabulary building, term frequency scaling, and IDF weighting occur
   STRICTLY on the training split, completely eliminating test set leakage.
2. Stratified Train/Test Split: Preserves exact class balance across categories.
3. Metric Suite: Computes Accuracy, Macro & Weighted Precision, Recall, and F1
   to evaluate balanced performance across all job categories.
4. Feature Attribution: Calculates linear log-odds contributions (TF-IDF value * coefficient)
   to explain predictions without relying on black-box approximations.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def build_classification_pipeline(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end sklearn Pipeline.
    Pipeline guarantees no data leakage: vectorizer learns IDF weights and vocabulary
    only from the training fold during fit().
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=ngram_range,
            sublinear_tf=True  # Applies 1 + log(tf) scaling to dampen overly frequent terms
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Guards against subtle class imbalances
            random_state=random_state,
            solver='lbfgs'
        ))
    ])


def train_and_evaluate_classifier(
    df: pd.DataFrame,
    text_col: str = 'job_description',
    label_col: str = 'category',
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5
) -> Tuple[Pipeline, Dict[str, Any], List[str]]:
    """
    Splits data stratifiably, fits the pipeline on training data ONLY,
    and calculates comprehensive evaluation metrics on held-out test data.

    Returns:
        pipeline: Trained sklearn Pipeline
        metrics: Dictionary containing accuracy, macro/weighted precision, recall, f1, and confusion matrix
        class_names: Alphabetically sorted list of target classes
    """
    clean_df = df.dropna(subset=[text_col, label_col]).copy()
    X = clean_df[text_col].astype(str).tolist()
    y = clean_df[label_col].astype(str).tolist()

    class_names = sorted(list(set(y)))

    # Stratified split ensures equal distribution of all 5 job categories in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = build_classification_pipeline(random_state=random_state)

    # Cross-validation on training data to assess variance before testing
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_macro')

    # Fit ONLY on training data
    pipeline.fit(X_train, y_train)

    # Predict on held-out test split
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Compute comprehensive evaluation metrics
    conf_mat = confusion_matrix(y_test, y_pred, labels=class_names)
    
    metrics: Dict[str, Any] = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'cv_f1_macro_mean': float(np.mean(cv_scores)),
        'cv_f1_macro_std': float(np.std(cv_scores)),
        'confusion_matrix': conf_mat.tolist(),
        'test_size': len(y_test),
        'train_size': len(y_train),
        'classification_report': classification_report(y_test, y_pred, labels=class_names, output_dict=True, zero_division=0)
    }

    return pipeline, metrics, class_names


def predict_category(
    pipeline: Pipeline,
    text: str
) -> Tuple[str, Dict[str, float]]:
    """
    Predicts the category of a given text and outputs calibrated class probabilities.
    """
    pred_label = str(pipeline.predict([text])[0])
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    prob_dict = {
        str(cls_name): round(float(prob) * 100, 2)
        for cls_name, prob in zip(classes, probabilities)
    }
    # Sort descending by probability
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    return pred_label, sorted_probs


def explain_prediction(
    pipeline: Pipeline,
    text: str,
    target_class: str = None,
    top_n: int = 8
) -> List[Tuple[str, float]]:
    """
    Generates explainable feature attributions for a given prediction.

    Mathematical Basis:
    In multinomial logistic regression, the log-odds score for class k is:
        z_k = beta_0,k + sum_j (beta_j,k * x_j)
    The contribution of feature j to class k is (beta_j,k * x_j).
    Positive contributions increase the log-odds toward class k.

    Returns:
        List of tuples: (term, contribution_score) sorted descending by contribution.
    """
    vectorizer: TfidfVectorizer = pipeline.named_steps['tfidf']
    clf: LogisticRegression = pipeline.named_steps['clf']

    classes = list(clf.classes_)
    if target_class is None:
        target_class = str(pipeline.predict([text])[0])

    if target_class not in classes:
        return []

    class_idx = classes.index(target_class)

    # Vectorize single document
    vec = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get non-zero indices for this document
    nonzero_indices = vec.indices
    nonzero_values = vec.data

    if len(nonzero_indices) == 0:
        return []

    # Get coefficients for target class
    class_coefs = clf.coef_[class_idx]

    # Compute contribution = value * coefficient
    contributions = []
    for idx, val in zip(nonzero_indices, nonzero_values):
        term = feature_names[idx]
        coef = class_coefs[idx]
        contrib = float(val * coef)
        if contrib > 0:  # Positively contributed to selecting this class
            contributions.append((term, round(contrib, 4)))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:top_n]
