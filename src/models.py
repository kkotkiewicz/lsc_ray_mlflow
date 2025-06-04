from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "LogisticRegression": [
            {"name": "LogisticRegression_Default", "model": LogisticRegression(max_iter=1000, random_state=42)},
            {"name": "LogisticRegression_MaxIter2000", "model": LogisticRegression(max_iter=2000, random_state=42)},
            {"name": "LogisticRegression_L1_Saga_C01", "model": LogisticRegression(penalty='l1', solver='saga', max_iter=2000, C=0.1, random_state=42)},
            {"name": "LogisticRegression_L2_Liblinear_C10", "model": LogisticRegression(penalty='l2', solver='liblinear', C=1.0, random_state=42)},
        ],
        "RandomForest": [
            {"name": "RandomForest_Default", "model": RandomForestClassifier(n_estimators=100, random_state=42)},
            {"name": "RandomForest_N200", "model": RandomForestClassifier(n_estimators=200, random_state=42)},
            {"name": "RandomForest_MaxDepth10", "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)},
            {"name": "RandomForest_MinSamplesSplit10", "model": RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)},
        ],
        "SVM": [
            {"name": "SVM_Default", "model": SVC(probability=True, random_state=42)},
            {"name": "SVM_Linear_C01", "model": SVC(kernel='linear', C=0.1, probability=True, random_state=42)},
            {"name": "SVM_C10", "model": SVC(C=10, probability=True, random_state=42)},
            {"name": "SVM_Poly", "model": SVC(kernel='poly', degree=3, probability=True, random_state=42)},
        ]
    }