from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, session
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
from datetime import datetime

# Import custom modules
from models.data_processor import DataProcessor
from models.ml_models import ModelTrainer
from models.visualizations import create_sleep_duration_plot, create_sleep_quality_plot, create_physical_activity_plot
from models.utils import allowed_file, save_uploaded_file
from models.visualizations import create_sleep_disorder_distribution_plot

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize global variables
data_processor = None
model_trainer = None
results = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/dashboard')
# def dashboard():
#      if 'data_loaded' not in session or not session['data_loaded']:
#          flash('Please upload data first', 'warning')
#          return redirect(url_for('index'))
    
#      return render_template('dashboard.html', 
#                            results=results,
#                            sleep_duration_by_occupation=results.get('sleep_duration_plot', ''),
#                            sleep_quality_by_occupation=results.get('sleep_quality_plot', ''),
#                            physical_activity_plot=results.get('physical_activity_plot', ''),
#                            sleep_disorder_distribution=results.get('sleep_disorder_distribution', ''),
#                            model_performance=results.get('model_performance', {}))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
     if 'data_loaded' not in session or not session['data_loaded']:
         flash('Please upload data first', 'warning')
         return redirect(url_for('index'))

     # Get selected plot from dropdown (default to sleep_duration)
     selected_plot_type = request.form.get('viz_select', 'sleep_duration')
     y_true, y_pred = None, None
     if selected_plot_type == 'confusion_matrix':
        y_true, y_pred = model_trainer.get_validation_results()  # define this method
     # Import the dispatcher (if not already)
     from models.visualizations import generate_plot  # You must define this dispatcher function

     selected_plot_image = generate_plot(selected_plot_type, data_processor.processed_data,y_true, y_pred)
     print("Plot selected:", selected_plot_type)
     print("Columns in data:", data_processor.processed_data.columns.tolist())

     return render_template('dashboard.html',
                            results=results,
                            sleep_duration_by_occupation=results.get('sleep_duration_plot', ''),
                            sleep_quality_by_occupation=results.get('sleep_quality_plot', ''),
                            physical_activity_plot=results.get('physical_activity_plot', ''),
                            sleep_disorder_distribution=results.get('sleep_disorder_distribution', ''),
                            user_selected_plot=selected_plot_image,
                            selected_plot_type=selected_plot_type)


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Get form data and make prediction
#         form_data = request.form.to_dict()
        
#         # Convert form data to DataFrame format expected by model
#         input_df = pd.DataFrame([form_data])
        
#         # Ensure model trainer is initialized
#         if model_trainer is None:
#             flash('Model not trained. Please upload and process data first', 'danger')
#             return redirect(url_for('predict'))
            
#         # Make prediction
#         prediction, probability = model_trainer.predict(input_df)
        
#         return render_template('prediction_result.html', 
#                               prediction=prediction,
#                               probability=probability,
#                               input_data=form_data)
    
#     return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict()

        # Define required columns (as expected by your model)
        required_columns = ['Heart Rate', 'Daily Steps', 'BMI Category', 'Stress Level',
                            'Blood Pressure', 'Sleep Duration', 'Quality of Sleep',
                            'Person ID', 'Physical Activity Level']

        # Fill missing fields with default values
        for col in required_columns:
            if col not in form_data:
                form_data[col] = '0'  # or some default, e.g., 'Unknown', 'Normal', etc.

        # Create DataFrame
        input_df = pd.DataFrame([form_data])

        # Ensure numeric columns are converted properly
        numeric_columns = ['Heart Rate', 'Daily Steps', 'Sleep Duration']
        for col in numeric_columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Check if model is initialized
        if model_trainer is None:
            flash('Model not trained. Please upload and process data first', 'danger')
            return redirect(url_for('predict'))

        try:
            # Make prediction
            prediction, probability = model_trainer.predict(input_df)

            return render_template('prediction_result.html', 
                                   prediction=prediction,
                                   probability=probability,
                                   input_data=form_data)
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'danger')
            return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global data_processor, model_trainer, results
    
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
        filepath = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        
        try:
            # Process data
            data_processor = DataProcessor(filepath)
            cleaned_data = data_processor.clean_data()
            
            # Train models
            model_trainer = ModelTrainer(cleaned_data)
            model_trainer.prepare_data()
            model_performance = model_trainer.train_and_evaluate()
            
           

            # Generate visualizations
            results['sleep_duration_plot'] = create_sleep_duration_plot(cleaned_data)
            results['sleep_quality_plot'] = create_sleep_quality_plot(cleaned_data)
            results['physical_activity_plot'] = create_physical_activity_plot(cleaned_data)
            results['model_performance'] = model_performance
            results['sleep_disorder_distribution'] = create_sleep_disorder_distribution_plot(cleaned_data)
            # Set session flag
            
            # Set session flag
            session['data_loaded'] = True
            
            flash('Data successfully processed', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload CSV or Excel file', 'danger')
        return redirect(url_for('index'))

# @app.route('/download_report')
# def download_report():
#     if data_processor is None or model_trainer is None:
#         flash('No data to generate report', 'danger')
#         return redirect(url_for('dashboard'))
        
#     # Generate report logic
#     # ...
    
    # return "Report generation functionality will be implemented here"
    

@app.route('/download_report')
def download_report():
    # Replace this path with the actual path to your IEEE paper PDF
    return send_file(r'C:\Users\vijaya shree R\OneDrive\DSA6114\Project_ML_sleepdisorder\project_2\final_paper_DSA.pdf', as_attachment=True)

@app.route('/disorder-distribution')
def disorder_distribution():
    data = load_your_processed_data()
    image_str = create_sleep_disorder_distribution_plot(data)
    return render_template('your_template.html', image_str=image_str)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)