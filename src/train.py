import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


def train_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # Save model
    with open('src/models/iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train_model()
    print("Model trained and saved successfully!")
    