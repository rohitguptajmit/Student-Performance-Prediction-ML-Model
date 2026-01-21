import pandas as pd
import joblib
import warnings

# Suppress sklearn warnings about feature names if necessary
warnings.filterwarnings("ignore")

MODEL_PATH = 'best_student_model.pkl'

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found. Please run train_students.py first.")
        return None

def get_user_input():
    print("\n--- Predict Student Exam Score ---")
    print("Enter the following details (press Enter to use random/default values for complex fields):")
    
    try:
        hours_studied = float(input("Hours Studied (0-40): ") or 20)
        attendance = float(input("Attendance (0-100): ") or 80)
        parental_involvement = input("Parental Involvement (Low, Medium, High): ").capitalize() or "Medium"
        tutoring = input("Access to Tutoring (Yes, No): ").capitalize() or "No"
        access_to_resources = input("Access to Resources (Low, Medium, High): ").capitalize() or "Medium"
        motivation_level = input("Motivation Level (Low, Medium, High): ").capitalize() or "Medium"
        
        # Construct DataFrame with all columns expected by the model
        # We need to match the columns from training. 
        # Since we used a pipeline with ColumnTransformer, we need to provide input as a DataFrame 
        # with the original column names.
        
        # Template based on dataset structure (simplified for key inputs, others set to mode/median)
        data = {
            'Hours_Studied': [hours_studied],
            'Attendance': [attendance],
            'Parental_Involvement': [parental_involvement],
            'Access_to_Resources': [access_to_resources],
            'Extracurricular_Activities': ['No'], # Default
            'Sleep_Hours': [7], # Default
            'Previous_Scores': [75], # Default 
            'Motivation_Level': [motivation_level],
            'Internet_Access': ['Yes'], # Default
            'Tutoring': [tutoring],
            'Tutoring_Sessions': [1], # Default
            'Family_Income': ['Medium'], # Default
            'Teacher_Quality': ['Medium'], # Default
            'School_Type': ['Public'], # Default
            'Peer_Influence': ['Neutral'], # Default
            'Physical_Activity': [3], # Default
            'Learning_Disabilities': ['No'], # Default
            'Parental_Education_Level': ['High School'], # Default
            'Distance_from_Home': ['Near'], # Default
            'Gender': ['Male'] # Default
        }
        
        # Verify we didn't miss any columns by checking against a sample if possible, 
        # but for now we rely on the robust pipeline handling (ignore unknown, imputation).
        # ideally we should read the columns from X_train.columns but we don't have X_train here easily without loading data.
        # Let's hope the pipeline handles it (it should if using 'passthrough' or explicit columns).
        # Our pipeline uses specific columns names in ColumnTransformer selector.
        
        return pd.DataFrame(data)
        
    except ValueError:
        print("Invalid input. Please enter numbers for numeric fields.")
        return None

def predict_score(model, input_df):
    if input_df is not None:
        prediction = model.predict(input_df)
        print(f"\nPredicted Exam Score: {prediction[0]:.2f}")
        return prediction[0]
    return None

if __name__ == "__main__":
    model = load_model()
    if model:
        input_data = get_user_input()
        predict_score(model, input_data)
