import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import io
import base64
import matplotlib
import plotly.express as px

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

import random
import nltk
from nltk.chat.util import Chat, reflections

from collections import defaultdict
import random
import json
import os

from sklearn.linear_model import RidgeClassifier



nltk.download("punkt")

# Use Agg backend to avoid GUI issues
matplotlib.use('Agg')

# Load dataset
df = pd.read_excel("Sleep_health_and_lifestyle_dataset.xlsx")
df.columns = df.columns.str.strip()
df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Other')

# Handle mixed or string-like numeric values
def convert_to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

# Handle missing values
numeric_cols = df.select_dtypes(include=['int', 'float']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Encode categorical variables
label_enc = LabelEncoder()
df['Gender'] = label_enc.fit_transform(df['Gender'].astype(str))
df['Occupation'] = label_enc.fit_transform(df['Occupation'].astype(str))
print(df['BMI Category'].value_counts())
df['BMI Category'] = label_enc.fit_transform(df['BMI Category'].astype(str))
print(df['BMI Category'].value_counts())
df['Sleep Disorder'] = df['Sleep Disorder'].replace({'Other': 3, 'Sleep Apnea': 1, 'Insomnia': 2}).astype(int)

# Feature scaling
features = ['Age', 'Sleep Duration', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Dimensionality reduction
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df[features])
X = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
y = df['Sleep Disorder']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
svm_model = SVC().fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
k_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42).fit(X_train, y_train)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

ridge_model = RidgeClassifier().fit(X_train, y_train)

# Evaluating Random Forest Model
print("\nRandom Forest Performance:")
rf_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Evaluating SVM Model
print("\nSVM Model Performance:")
svm_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Evaluating KNN Model
print("\nKNN Model Performance:")
k_pred = k_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, k_pred))
print(classification_report(y_test, k_pred))


# ➕ Evaluation for Neural Network
print("\nNeural Network Performance:")
nn_pred = nn_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, nn_pred))
print(classification_report(y_test, nn_pred))

# ➕ Evaluation for Gradient Boosting
print("\nGradient Boosting Performance:")
gb_pred = gb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))

print("\nRidge Classifier Performance:")
ridge_pred = ridge_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ridge_pred))
print(classification_report(y_test, ridge_pred))


# Define possible sleep improvement actions
ACTIONS = [
    "Maintain a consistent sleep schedule.",
    "Avoid caffeine before bedtime.",
    "Reduce screen time 1 hour before sleep.",
    "Engage in physical activity daily.",
    "Try meditation or relaxation techniques.",
    "Maintain a balanced diet for better sleep.",
    "Use dim lights before sleeping.",
    "Improve room temperature for optimal sleep.",
]

# Q-Table: Store action values for each state
Q_TABLE_FILE = "q_table.json"

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.q_table = self.load_q_table()  # Load past Q-table

    def load_q_table(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE, "r") as f:
                return json.load(f)
        return defaultdict(lambda: {action: 0 for action in self.actions})

    def save_q_table(self):
        with open(Q_TABLE_FILE, "w") as f:
            json.dump(self.q_table, f)

    def get_state(self, data):
        """ Define state based on user sleep patterns. """
        return f"sleep_{data['sleep_duration']}_stress_{data['stress_level']}_activity_{data['daily_steps']}"

    def choose_action(self, state):
        """ Choose an action using ε-greedy policy. """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Exploration
        return max(self.q_table[state], key=self.q_table[state].get)  # Exploitation

    def update_q_value(self, state, action, reward):
        """ Update Q-table based on feedback. """
        best_next_action = max(self.q_table[state], key=self.q_table[state].get)
        self.q_table[state][action] = self.q_table[state][action] + self.alpha * (
            reward + self.gamma * self.q_table[state][best_next_action] - self.q_table[state][action]
        )
        self.save_q_table()  # Save updated Q-table
agent = QLearningAgent(ACTIONS)  # Initialize RL agent

# Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def calculate_sleep_score(data):
    score = 100
    if float(data["sleep_duration"]) < 6:
        score -= 30
    if float(data["stress_level"]) > 7:
        score -= 20
    if float(data["bmi"]) > 25:
        score -= 15
    if float(data["daily_steps"]) < 5000:
        score -= 10
    return max(score, 0)  # Ensure score is not negative

@app.route("/sleep_score", methods=["POST"])
def sleep_score():
    data = request.form
    score = calculate_sleep_score(data)
    return jsonify({"score": score})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.form
    input_data = np.array([
        float(data["age"]), float(data["sleep_duration"]), float(data["stress_level"]),
        float(data["bmi"]), float(data["heart_rate"]), float(data["daily_steps"])
    ]).reshape(1, -1)
    
    scaled_data = scaler.transform(input_data)
    pca_data = pca.transform(scaled_data)
    # ➕ Predict from all models
    predictions = {
        "SVM": svm_model.predict(pca_data)[0],
        "Random Forest": rf_model.predict(pca_data)[0],
        "KNN": k_model.predict(pca_data)[0],
        "Neural Network": nn_model.predict(pca_data)[0],
        "Gradient Boosting": gb_model.predict(pca_data)[0],
        "Ridge Classifier": ridge_model.predict(pca_data)[0] 
    }

    
    disorder_types = {0: "None", 1: "Sleep Apnea", 2: "Insomnia", 3: "Other"}
    # predicted_disorder = disorder_types.get(prediction, "Unknown")
    results = {k: disorder_types.get(v, "Unknown") for k, v in predictions.items()}

    # Generate personalized recommendations
    recommendations = []
    
    if float(data["sleep_duration"]) < 6:
        recommendations.append("Try to get at least 7-8 hours of sleep for better health.")
    if float(data["stress_level"]) > 7:
        recommendations.append("Practice meditation or relaxation techniques to reduce stress.")
    if float(data["bmi"]) > 25:
        recommendations.append("Maintain a balanced diet and regular exercise to improve sleep quality.")
    if float(data["daily_steps"]) < 5000:
        recommendations.append("Increase daily physical activity for better sleep.")

    return jsonify({
        "classification": results,
        "recommendations": recommendations
    })

@app.route("/personalized_recommendation", methods=["POST"])
def personalized_recommendation():
    data = request.form
    state = agent.get_state(data)
    action = agent.choose_action(state)
    return jsonify({
        "recommendation": action,
        "message": "Based on your sleep habits, try this recommendation."
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.form
    state = agent.get_state(data)
    action = data["action"]
    reward = int(data["reward"])  # User rates recommendation (e.g., -1 = bad, 1 = good)
    
    agent.update_q_value(state, action, reward)
    
    return jsonify({
        "message": "Thank you for your feedback! The system is learning to give better recommendations."
    })


@app.route("/visualization")
def visualization():
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sleep Disorder', data=df)
    plt.title("Distribution of Sleep Disorders")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url1 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    sns.pairplot(df)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url2 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True)
    plt.title("Correlation Heatmap")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url3 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # # Violin Plot
    # plt.figure(figsize=(8, 6))
    # sns.violinplot(x='Sleep Disorder', y='Stress Level', data=df)
    # plt.title("Violin Plot of Stress Level by Sleep Disorder")
    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # img.seek(0)
    # plot_url4 = base64.b64encode(img.getvalue()).decode()
    # plt.close()
    # Violin Plot
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Sleep Disorder', y='Stress Level', data=df)
    plt.title("Violin Plot of Stress Level by Sleep Disorder")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url4 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Sleep Duration', kde=True)
    plt.title("Sleep Duration Histogram")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url5 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    
    # Accuracy scores
   
   # 🆕 Predictions for accuracy (keep these separate from cross_val_predict)
    svm_pred_test = svm_model.predict(X_test)
    rf_pred_test = rf_model.predict(X_test)
    k_pred_test = k_model.predict(X_test)
    nn_pred_test = nn_model.predict(X_test)
    gb_pred_test = gb_model.predict(X_test)

    # ✅ Accuracy scores using correct predictions
    svm_acc = accuracy_score(y_test, svm_pred_test)
    rf_acc = accuracy_score(y_test, rf_pred_test)
    k_acc = accuracy_score(y_test, k_pred_test)
    nn_acc = accuracy_score(y_test, nn_pred_test)
    gb_acc = accuracy_score(y_test, gb_pred_test)


    # Create accuracy plot
    model_names = ['SVM', 'Random Forest', 'KNN', 'Neural Network', 'Gradient Boosting']
    accuracy_scores = [svm_acc, rf_acc, k_acc, nn_acc, gb_acc]                                          

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=accuracy_scores, palette="coolwarm")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    for index, value in enumerate(accuracy_scores):
        plt.text(index, value + 0.01, f"{value:.2f}", ha='center')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    accuracy_plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Classification Reports
    kf = KFold(n_splits=5)
    svm_pred = cross_val_predict(svm_model, X, y, cv=kf)
    rf_pred = cross_val_predict(rf_model, X, y, cv=kf)
    nn_pred = cross_val_predict(nn_model, X, y, cv=kf)
    gb_pred = cross_val_predict(gb_model, X, y, cv=kf)
    svm_report = classification_report(y, svm_pred)
    rf_report = classification_report(y, rf_pred)
    nn_report = classification_report(y, nn_pred)
    gb_report = classification_report(y, gb_pred)
    

    return jsonify({
        "plot_url1": plot_url1,
        "plot_url2": plot_url2,
        "plot_url3": plot_url3,
        "plot_url4": plot_url4,
        "plot_url5": plot_url5,
        "svm_report": svm_report,
        "rf_report": rf_report,
        "nn_report": nn_report,
        "gb_report": gb_report,
        "accuracy_plot": accuracy_plot_url  # 🔥 New plot
    })

pairs = [
    (r"(hello|hi|hey)", ["Hello! How can I assist you with your sleep concerns?"]),
    (r"how can i (improve|fix|manage) my sleep", 
     ["Maintain a consistent sleep schedule, avoid caffeine before bed, and create a relaxing bedtime routine."]),
    (r"what is (insomnia|sleep apnea|narcolepsy)", 
     ["%1 is a sleep disorder that affects sleep quality and overall health. Would you like more information?"]),
    (r"how much sleep do I need", 
     ["Adults typically need 7-9 hours of sleep per night for optimal health."]),
    (r"does stress affect sleep", 
     ["Yes, high stress levels can lead to poor sleep. Try meditation, exercise, and relaxation techniques."]),
    (r"what should I do if I can't sleep", 
     ["Try deep breathing exercises, avoid screen time before bed, and create a comfortable sleeping environment."]),
    (r"thank you", ["You're welcome! Sleep well!"]),
    (r"see my sleep score", ["Let me analyze your sleep score. If it's low, try improving sleep hygiene."]),
    (r"(.*)", ["I don't fully understand, but I can provide general sleep health advice!"])
]

chatbot = Chat(pairs, reflections)

@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message").lower()
    bot_response = chatbot.respond(user_input)
    
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)

