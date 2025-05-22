💤 Sleep Disorder Prediction using Machine Learning
 📌 Overview

Sleep disorders affect millions worldwide, significantly impacting health and quality of life. Early detection can reduce medical risks and healthcare costs. This project presents a data-driven approach to predict sleep disorders using the **Sleep Health and Lifestyle Dataset**.

The system classifies individuals into one of the following categories:

- **None**
- **Insomnia**
- **Sleep Apnea**

 📊 Dataset

The dataset contains **400 records** with the following features:

- Age
- Occupation
- Sleep Duration
- Physical Activity Level
- ...and more lifestyle-related variables

 🧠 Methodology

Our method includes:

- Data Cleaning & Preprocessing
- Feature Selection
- Model Training & Evaluation

We evaluated the following machine learning models:

- Logistic Regression ✅ *(Best Performing)*
- Linear Support Vector Machine
- Random Forest
- Gradient Boosting

**Logistic Regression achieved the highest accuracy**, making it the most effective model for this dataset.

📈 Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

✅ Results

The results demonstrate that machine learning algorithms can effectively predict sleep disorders using lifestyle and demographic data. This approach has potential applications in:

- Early screening for sleep disorders
- Clinical decision support systems
- Insurance risk assessments
 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- Matplotlib / Seaborn (for visualizations)
- Flask (for Web App Integration)
 🚀 How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/your-username/SleepDisorder_Prediction.git
   cd SleepDisorder_Prediction
2.Install dependencies

  pip install -r requirements.txt

3.Run the Flask app
  python app.py
  
4.Open your browser and go to:
  
  http://127.0.0.1:5000

📌 Future Enhancements

  Integrate a chatbot for patient interaction
  Include Q-learning for adaptive recommendation systems
  Add radar charts for visualizing model comparisons
