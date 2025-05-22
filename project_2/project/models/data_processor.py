import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataProcessor:
    """
    Class for processing sleep disorder data.
    Handles data loading, cleaning, and transformation.
    """
    
    def __init__(self, file_path):
        """
        Initialize with the path to the data file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV or Excel file containing sleep data
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.load_data()
        
    def load_data(self):
        """Load data from the specified file path."""
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.raw_data = pd.read_csv(self.file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.raw_data = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")
                
            print(f"Data loaded successfully with {self.raw_data.shape[0]} rows and {self.raw_data.shape[1]} columns")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self):
        """
        Clean the raw data by handling missing values,
        removing duplicates, and correcting data types.
        
        Returns:
        --------
        pandas.DataFrame
            Cleaned data
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Create a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=['number']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with mean
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].mean())
            
        # Fill categorical columns with mode
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert data types if needed
        # Example: convert string dates to datetime
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                print(f"Could not convert {col} to datetime")
        
        # Store the processed data
        self.processed_data = df
        
        print(f"Data cleaning complete. {df.shape[0]} rows remaining.")
        return df
    
    def extract_features(self):
        """
        Extract and engineer features from the cleaned data.
        
        Returns:
        --------
        pandas.DataFrame
            Data with engineered features
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please clean data first.")
            
        df = self.processed_data.copy()
        
        # Example feature engineering
        # Calculate BMI if height and weight are available
        if 'Height' in df.columns and 'Weight' in df.columns:
            df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
            
        # Create age categories if Age is available
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(
                df['Age'], 
                bins=[0, 18, 35, 50, 65, 100],
                labels=['Under 18', '18-35', '36-50', '51-65', 'Over 65']
            )
            
        # Create sleep duration categories if Sleep_Duration is available
        if 'Sleep_Duration' in df.columns:
            df['Sleep_Category'] = pd.cut(
                df['Sleep_Duration'],
                bins=[0, 6, 7.5, 9, 24],
                labels=['Poor', 'Normal', 'Good', 'Excessive']
            )
        
        return df
    
    def get_data_summary(self):
        """
        Generate a summary of the processed data.
        
        Returns:
        --------
        dict
            Summary statistics and information about the dataset
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Please clean data first.")
            
        df = self.processed_data
        
        summary = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'column_names': df.columns.tolist(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'descriptive_stats': df.describe().to_dict()
        }
        
        return summary