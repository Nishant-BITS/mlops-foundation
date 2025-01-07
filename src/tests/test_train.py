import os
import pickle
from src.train import train_model


def test_model_training():
    # Train the model
    train_model()
    
    # Check if model file exists
    assert os.path.exists('src/models/iris_model.pkl'), "Model file not found!"

    # Check if the file is a valid pickle
    with open('src/models/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None, "Model is not a valid pickle object!"

    print("All tests passed!")