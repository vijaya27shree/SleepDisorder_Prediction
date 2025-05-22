import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from io import BytesIO

class ModelTrainer:
    """
    Class for training and evaluating machine learning models for sleep disorder prediction.
    """
    
    def __init__(self, data):
        """
        Initialize the ModelTrainer with a cleaned dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Cleaned sleep disorder dataset
        """
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.preprocessor = None
        
        # Create models directory if it doesn't exist
        os.makedirs('models/saved_models', exist_ok=True)
    
    def prepare_data(self, target_column='Sleep_Disorder', test_size=0.2, random_state=42):
        """
        Prepare data for model training by splitting into train/test sets.
        
        Parameters:
        -----------
        target_column : str
            Name of the column to predict
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        # Create X and y
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Set up preprocessing
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Define preprocessing for numerical and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        print(f"Data prepared for training with {len(self.X_train)} training samples and {len(self.X_test)} test samples")
    
    # def train_and_evaluate(self):
    #     """
    #     Train and evaluate multiple models on the prepared data.
        
    #     Returns:
    #     --------
    #     dict
    #         Performance metrics for each model
    #     """
    #     self.X_test = X_test
    #     self.y_test = y_test
    #     self.model = model

    #     if self.X_train is None or self.y_train is None:
    #         raise ValueError("Data not prepared. Call prepare_data() first.")
            
    #     # Define models to train
    #     models_to_train = {
    #         'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    #         'Linear SVC': LinearSVC(random_state=42),
    #         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    #         'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    #     }
        
    #     results = {}
    #     best_accuracy = 0
        
    #     # Train each model
    #     for name, model in models_to_train.items():
    #         # Create pipeline with preprocessing
    #         pipeline = Pipeline(steps=[
    #             ('preprocessor', self.preprocessor),
    #             ('model', model)
    #         ])
            
    #         # Train the model
    #         pipeline.fit(self.X_train, self.y_train)
            
    #         # Make predictions
    #         y_pred = pipeline.predict(self.X_test)
            
    #         # Calculate metrics
    #         accuracy = accuracy_score(self.y_test, y_pred)
    #         precision = precision_score(self.y_test, y_pred, average='weighted')
    #         recall = recall_score(self.y_test, y_pred, average='weighted')
    #         f1 = f1_score(self.y_test, y_pred, average='weighted')
    #         conf_matrix = confusion_matrix(self.y_test, y_pred).tolist()
            
    #         # Store results
    #         results[name] = {
    #             'accuracy': round(accuracy, 4),
    #             'precision': round(precision, 4),
    #             'recall': round(recall, 4),
    #             'f1_score': round(f1, 4),
    #             'confusion_matrix': conf_matrix
    #         }
            
    #         # Save the model
    #         self.models[name] = pipeline
            
    #         # Track best model
    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             self.best_model = name
                
    #             # Save the best model
    #             joblib.dump(pipeline, f'models/saved_models/best_model.pkl')
                
    #     # Save model performance data
    #     results['best_model'] = self.best_model
        
    #     print(f"Model training complete. Best model: {self.best_model} with accuracy: {best_accuracy:.4f}")
    #     return results
    
    
    # def get_validation_results(self):
    #     """
    #     Return ground truth and predictions on the validation set
    #     """
    #     if hasattr(self, 'X_test') and hasattr(self, 'y_test') and hasattr(self, 'model'):
    #         y_pred = self.model.predict(self.X_test)
    #         return self.y_test, y_pred
    #     else:
    #         raise AttributeError("Validation data or model not found. Make sure to train the model first.")
   
    def train_and_evaluate(self):
        """
        Train and evaluate multiple models on the prepared data.
        
        Returns:
        --------
        dict
            Performance metrics for each model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVC': LinearSVC(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        best_accuracy = 0
        
        # Train each model
        for name, model in models_to_train.items():
            # Create pipeline with preprocessing
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            
            # Train the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(self.y_test, y_pred).tolist()
            
            # Store results
            results[name] = {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'confusion_matrix': conf_matrix
            }
            
            # Save the model
            self.models[name] = pipeline
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = name

                # Save the best model
                joblib.dump(pipeline, f'models/saved_models/best_model.pkl')

                # ✅ Save the test data and model for future use (for confusion matrix)
                self.model = pipeline
                self.X_test_saved = self.X_test
                self.y_test_saved = self.y_test
        
        # Save model performance data
        results['best_model'] = self.best_model
        
        print(f"Model training complete. Best model: {self.best_model} with accuracy: {best_accuracy:.4f}")
        return results

    def get_validation_results(self):
        """
        Returns y_test and predictions using the best trained model.
        """
        if hasattr(self, 'X_test_saved') and hasattr(self, 'y_test_saved') and hasattr(self, 'model'):
            y_pred = self.model.predict(self.X_test_saved)
            return self.y_test_saved, y_pred
        else:
            raise AttributeError("Validation data or trained model is missing. Make sure training has completed.")

    def predict(self, input_data):
        """
        Make predictions on new data using the best model.
        
        Parameters:
        -----------
        input_data : pandas.DataFrame
            New data to make predictions on
            
        Returns:
        --------
        tuple
            (prediction, probability)
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call train_and_evaluate() first.")
            
        # Load the best model
        model = self.models[self.best_model]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(input_data)[0].max()
        except:
            # Some models like SVC don't have predict_proba
            probability = None
        
        return prediction, probability