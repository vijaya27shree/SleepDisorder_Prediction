import os
import pandas as pd
import numpy as np
from datetime import datetime
import random
import string
from werkzeug.utils import secure_filename

def allowed_file(filename, allowed_extensions):
    """
    Check if the file has an allowed extension
    
    Parameters:
    -----------
    filename : str
        Name of the file
    allowed_extensions : set
        Set of allowed file extensions
        
    Returns:
    --------
    bool
        True if file is allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_uploaded_file(file, upload_folder):
    """
    Save the uploaded file to the upload folder
    
    Parameters:
    -----------
    file : FileStorage
        Uploaded file object
    upload_folder : str
        Path to upload folder
        
    Returns:
    --------
    str
        Path to the saved file
    """
    # Generate a unique filename to avoid conflicts
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    unique_filename = f"{timestamp}_{random_string}_{filename}"
    
    # Save the file
    filepath = os.path.join(upload_folder, unique_filename)
    file.save(filepath)
    
    return filepath

def generate_sample_data(num_samples=100, output_path='sample_sleep_data.csv'):
    """
    Generate sample sleep data for testing
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    output_path : str
        Path to save the generated data
        
    Returns:
    --------
    pandas.DataFrame
        Generated sample data
    """
    # Define possible values for categorical columns
    genders = ['Male', 'Female']
    occupations = ['Healthcare', 'Engineer', 'Teacher', 'Accountant', 'Lawyer', 
                  'Salesperson', 'Doctor', 'Nurse', 'Manager', 'Artist']
    bmi_categories = ['Normal', 'Overweight', 'Obese', 'Underweight']
    disorders = ['None', 'Insomnia', 'Sleep Apnea', None, None, None]  # More None values to simulate imbalance
    
    # Generate random data
    data = {
        'Person_ID': range(1, num_samples + 1),
        'Gender': [random.choice(genders) for _ in range(num_samples)],
        'Age': [random.randint(18, 80) for _ in range(num_samples)],
        'Occupation': [random.choice(occupations) for _ in range(num_samples)],
        'Sleep_Duration': [round(random.uniform(4, 10), 1) for _ in range(num_samples)],
        'Quality_of_Sleep': [random.randint(1, 10) for _ in range(num_samples)],
        'Physical_Activity': [random.randint(10, 90) for _ in range(num_samples)],
        'Stress_Level': [random.randint(1, 10) for _ in range(num_samples)],
        'BMI_Category': [random.choice(bmi_categories) for _ in range(num_samples)],
        'Heart_Rate': [random.randint(60, 100) for _ in range(num_samples)],
        'Daily_Steps': [random.randint(3000, 15000) for _ in range(num_samples)],
        'Sleep_Disorder': [random.choice(disorders) for _ in range(num_samples)]
    }
    
    # Add some correlations to make data more realistic
    for i in range(num_samples):
        # People with more physical activity tend to have better sleep quality
        if data['Physical_Activity'][i] > 60:
            data['Quality_of_Sleep'][i] = min(10, data['Quality_of_Sleep'][i] + random.randint(1, 3))
            
        # People with higher stress tend to have worse sleep quality
        if data['Stress_Level'][i] > 7:
            data['Quality_of_Sleep'][i] = max(1, data['Quality_of_Sleep'][i] - random.randint(1, 3))
            
        # Adjust sleep disorders based on sleep quality
        if data['Quality_of_Sleep'][i] < 4:
            data['Sleep_Disorder'][i] = 'Insomnia' if random.random() < 0.7 else 'Sleep Apnea'
        
        # Adjust sleep duration based on disorder
        if data['Sleep_Disorder'][i] == 'Insomnia':
            data['Sleep_Duration'][i] = max(4, data['Sleep_Duration'][i] - random.uniform(1, 2))
        elif data['Sleep_Disorder'][i] == 'Sleep Apnea':
            data['Sleep_Duration'][i] = max(4, data['Sleep_Duration'][i] - random.uniform(0.5, 1.5))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values
    for col in df.columns:
        if col not in ['Person_ID', 'Sleep_Disorder']:  # Don't add missing values to these columns
            mask = np.random.random(len(df)) < 0.05  # 5% missing values
            df.loc[mask, col] = np.nan
    
    # Save to CSV
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
    
    return df