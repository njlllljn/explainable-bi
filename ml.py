"""Small, auditable sklearn workflows used by the dashboard."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


def prepare_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert selected CSV columns to numeric values and impute missing data."""
    x = frame[columns].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=columns, index=x.index)


def numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert feature columns to numbers without fitting preprocessing on all rows."""
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def regression_results(frame: pd.DataFrame, features: list[str], target: str, alpha: float):
    x, y = numeric_features(frame, features), pd.to_numeric(frame[target], errors="coerce")
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
    models = {
        "Multiple Linear Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "Polynomial Regression (degree 2)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
        f"Ridge (α={alpha:g})": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)),
        f"Lasso (α={alpha:g})": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Lasso(alpha=alpha, max_iter=10000)),
        "Random Forest Regressor": make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    }
    scores, fitted = [], {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        scores.append({"Model": name, "R²": r2_score(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** .5, "MAE": mean_absolute_error(y_test, pred)})
        fitted[name] = model
    return pd.DataFrame(scores), fitted, (x_test, y_test)


def classification_results(frame: pd.DataFrame, features: list[str], target: str, c: float):
    x, y = numeric_features(frame, features), pd.to_numeric(frame[target], errors="coerce").dropna().astype(int)
    x = x.loc[y.index]
    observed_classes = set(y.unique())
    if observed_classes != {0, 1}:
        raise ValueError("classification target must contain both classes coded as 0 and 1")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42, stratify=y)
    models = {
        "Logistic Regression (L2)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=2000)),
        "SVM (RBF)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=c, probability=True, random_state=42)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)),
    }
    scores, fitted = [], {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1] if len(model.classes_) > 1 else pred
        scores.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1": f1_score(y_test, pred, zero_division=0), "ROC AUC": roc_auc_score(y_test, probability) if y_test.nunique() > 1 else np.nan})
        fitted[name] = model
    return pd.DataFrame(scores), fitted, (x_test, y_test)


def local_classification_explanation(model, row: pd.DataFrame, baseline: pd.Series,
                                     features: list[str], positive_class=1):
    """Show one-feature-at-a-time sensitivity around a selected record.

    This is a local model sensitivity check, not a causal effect or SHAP value.
    """
    classes = list(model.classes_)
    positive_index = classes.index(positive_class)
    current_probability = float(model.predict_proba(row)[0, positive_index])
    details = []
    for feature in features:
        typical_row = row.copy()
        typical_row.loc[:, feature] = baseline[feature]
        typical_probability = float(model.predict_proba(typical_row)[0, positive_index])
        details.append({
            "Feature": feature,
            "Selected value": float(row.iloc[0][feature]),
            "Typical value": float(baseline[feature]),
            "Risk probability at typical value": typical_probability,
            "Change from typical (percentage points)": (current_probability - typical_probability) * 100,
        })
    return current_probability, pd.DataFrame(details).sort_values(
        "Change from typical (percentage points)", key=lambda values: values.abs(), ascending=False
    )


def cluster_data(frame: pd.DataFrame, features: list[str], method: str, clusters: int):
    x = prepare_features(frame, features)
    scaled = StandardScaler().fit_transform(x)
    labels = KMeans(n_clusters=clusters, n_init=10, random_state=42).fit_predict(scaled) if method == "K-Means" else DBSCAN(eps=1.5, min_samples=8).fit_predict(scaled)
    points = PCA(n_components=2).fit_transform(scaled)
    return pd.DataFrame({"PC 1": points[:, 0], "PC 2": points[:, 1], "Segment": labels.astype(str)}, index=frame.index), pd.Series(labels, index=frame.index)


def feature_importance(model, features: list[str], x: pd.DataFrame, y: pd.Series, classification=False):
    """Permutation based held-out feature impact; robust across model families."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, x, y, n_repeats=5, random_state=42, n_jobs=-1,
                                    scoring="f1" if classification else "r2")
    return pd.DataFrame({"Feature": features, "Importance": result.importances_mean}).sort_values("Importance", ascending=False)


def q_learning_demo(episodes: int = 500, seed: int = 42):
    """Learn a small business policy in a transparent synthetic environment."""
    rng = np.random.default_rng(seed)
    actions = ["Increase marketing", "Maintain budget", "Offer targeted discount", "Improve retention"]
    # States: declining/steady/growing sales. Reward balances revenue lift and spend.
    rewards = np.array([
        [7.0, 1.0, 2.5, 5.0],
        [3.0, 2.0, 1.5, 4.5],
        [1.0, 3.0, -1.0, 2.0],
    ])
    transition = np.array([
        [[.58,.36,.06],[.25,.65,.10],[.35,.58,.07],[.22,.65,.13]],
        [[.20,.63,.17],[.08,.78,.14],[.18,.67,.15],[.12,.68,.20]],
        [[.08,.52,.40],[.05,.60,.35],[.10,.55,.35],[.06,.56,.38]],
    ])
    q = np.zeros((3, 4)); alpha, gamma, epsilon = .15, .85, .2
    for _ in range(episodes):
        state = 0
        for _ in range(20):
            action = rng.integers(4) if rng.random() < epsilon else int(q[state].argmax())
            next_state = int(rng.choice(3, p=transition[state, action]))
            reward = rewards[state, action] + rng.normal(0, .4)
            q[state, action] += alpha * (reward + gamma * q[next_state].max() - q[state, action])
            state = next_state
            if state == 2:
                break
    return q, actions, rewards
