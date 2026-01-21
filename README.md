# Student Performance Prediction ML Model

Welcome to the Student Performance Prediction project! This guide will help you understand, run, and experiment with a machine learning pipeline that predicts student exam scores based on factors like study hours, attendance, and parental involvement.

## � Data Source
The dataset used in this project is sourced from Kaggle:
[Student Performance Factors](https://www.kaggle.com/datasets/ayeshasiddiqa123/student-perfirmance)

## �📂 Project Structure

- **`train_students.py`**: The main script. It performs Data Analysis (EDA), preprocesses data, trains multiple models, evaluates them, and saves the best one.
- **`predict_score.py`**: An interactive script to make predictions using the trained model.
- **`StudentPerformanceAnalysis.ipynb`**: A Jupyter Notebook version of the analysis for interactive exploration.
- **`StudentPerformanceFactors.csv`**: The dataset used for training.
- **`output/`**: Directory containing generated charts (HTML) and performance metrics.
- **`best_student_model.pkl`**: The saved machine learning model (generated after running training).

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed (version 3.8 or higher is recommended).

You will need to install the following Python libraries:
```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib plotly joblib nbformat
```

### 2. Run the Analysis & Training
To analyze the data and train the models, run the following command in your terminal:

```bash
python train_students.py
```

**What this does:**
- loads the dataset.
- Generates interactive charts in the `output/` folder (Distribution, Correlation, etc.).
- Trains 6 different models (Linear, Ridge, Random Forest, XGBoost, etc.).
- Compares their performance and selects the best one (typically Ridge Regression).
- Saves the best model to `best_student_model.pkl`.

### 3. Explore the Results
Go to the `output/` folder to see:
- **`model_performance.csv`**: A table comparing MAE, RMSE, and R² scores.
- **`exam_score_dist.html`**: Interactive histogram of exam scores.
- **`correlation_matrix.html`**: Interactive heatmap of feature correlations.
- **`feature_importance.html`**: Bar chart showing which factors matter most (e.g., Hours Studied).

### 4. Make Predictions
Now that you have a trained model, you can predict scores for new students!

Run the prediction script:
```bash
python predict_score.py
```

Follow the on-screen prompts to enter details like:
- Hours Studied
- Attendance (%)
- Parental Involvement (Low, Medium, High)
- Access to Tutoring
- Motivation Level

The script will output a predicted **Exam Score**.

### 5. Jupyter Notebook (Optional)
If you prefer an interactive coding environment, open `StudentPerformanceAnalysis.ipynb` in VS Code or Jupyter Lab. 
- It contains the same logic as the training script but allows you to run code cell-by-cell and see charts inline.
- **Note**: Run the first cell `%pip install ...` to ensure your environment is set up correctly.

## 📊 Model Insights
Based on our analysis, the key drivers of high performance are:
1.  **Hours Studied**: There is a strong linear relationship between study time and scores.
2.  **Attendance**: Higher attendance consistently leads to better results.
3.  **Parental Involvement**: Active parental support correlates with higher scores.

The **Ridge Regression** model was chosen as the best performer, offering high accuracy (R² ~ 0.77) and good generalizability.

---
*Happy Learning!*
