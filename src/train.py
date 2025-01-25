import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def train_model():
    # Load data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    # Objective function for Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best Parameters:", study.best_params)

    # Save model
    best_model = RandomForestClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    with open('src/models/iris_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)


if __name__ == '__main__':
    train_model()
    print("Model trained and saved successfully!")
    