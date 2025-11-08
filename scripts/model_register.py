import os
import argparse
import mlflow

mlflow.start_run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained 
model")
    args = parser.parse_args()

    # Load the trained model from the provided path
    model = mlflow.sklearn.load_model(args.model)

    print("Registering the best trained used cars price prediction model")
    
    # Register the model in the MLflow Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_cars_price_prediction_model",
        artifact_path="random_forest_price_regressor"
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()
