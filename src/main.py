from preprocess import load_data
from models import get_models
from train import train_model

from sklearn.model_selection import train_test_split
import ray
import mlflow

ray.init(address="auto", ignore_reinit_error=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_groups = get_models()

futures = []
for group in model_groups.values():
    for entry in group:
        futures.append(train_model.remote(
            model_name=entry["name"],
            model=entry["model"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        ))

results = ray.get(futures)

print("\n=== Wyniki ===")
for model_name, acc, f1 in results:
    print(f"{model_name}: Accuracy={acc:.4f}, F1={f1:.4f}")
