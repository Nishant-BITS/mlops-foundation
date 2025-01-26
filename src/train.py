import mlflow.artifacts
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

import mlflow
import mlflow.sklearn

data = pd.read_csv('/Users/nishant/Documents/M.Tech Classes/SEM-3/MLOps/Assignment/Assignment-1/mlops-foundation/src/dataset/mobile_data.csv')
data = data.drop(['three_g'],axis=1)

x = data.iloc[:,:-1]
y = data['price_range']

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)


mlflow.set_experiment("mobile-price-classification-model")
with mlflow.start_run():
    random_state = 1
    max_depth=5
    min_samples_split=10

    mlflow.log_param("random_state",random_state)
    mlflow.log_param("max_depth",max_depth)
    mlflow.log_param("min_samples_split",min_samples_split)


    TreeModel = DecisionTreeRegressor(random_state=random_state,max_depth=max_depth,min_samples_split=min_samples_split)
    # Fit Model
    TreeModel.fit(X_train, Y_train)


    val_predictions = TreeModel.predict(X_test)
    val_mae = mean_absolute_error(val_predictions, Y_test)
    val_mse = mean_squared_error(val_predictions, Y_test)

    mlflow.log_metric("Mean absolute Error",val_mae)
    mlflow.log_metric("Mean Square error",val_mae)

    mlflow.sklearn.log_model(TreeModel,"MY_model")
    joblib.dump(TreeModel,"/Users/nishant/Documents/M.Tech Classes/SEM-3/MLOps/Assignment/Assignment-1/mlops-foundation/src/models/my_model.joblib")
    mlflow.log_artifact("/Users/nishant/Documents/M.Tech Classes/SEM-3/MLOps/Assignment/Assignment-1/mlops-foundation/src/models/MY_model.joblib")

    print("Validation MAE: {:,.0f}".format(val_mae))
    print("Validation MAE: {:,.0f}".format(val_mse))
    print("Prediction Result",val_predictions)
 


