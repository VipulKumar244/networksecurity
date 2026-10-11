import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data
)

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.main_utils.utils import evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        

    def track_mlflow(self,best_model,classificationmetric,model_name):
        with mlflow.start_run():
            mlflow.log_param("model", model_name)
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)
            mlflow.sklearn.log_model(best_model, name="model")

    def train_model(self,X_train,y_train,x_test,y_test):

        models = {
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(random_state=42),
            }
        
        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [50, 100]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.05],
                'subsample':[0.8],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [50, 100]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01],
                'n_estimators': [50, 100]
            }
            
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        if not model_report:
            raise ValueError("Model evaluation returned no scores")
        best_model_score = max(model_report.values())

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)


        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        inference_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", best_model),
        ])
        self.track_mlflow(inference_pipeline, classification_test_metric, best_model_name)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        network_model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=network_model)
        save_object("final_model/model.pkl",network_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact










    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)


            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
