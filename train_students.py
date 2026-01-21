import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Configuration
DATA_PATH = "StudentPerformanceFactors.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    return df

def perform_eda(df):
    print("\n--- EDA ---")
    
    # Summary Statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Distribution of Exam_Score
    fig_hist = px.histogram(df, x='Exam_Score', nbins=30, title='Distribution of Exam Score', marginal='box')
    fig_hist.write_html(os.path.join(OUTPUT_DIR, 'exam_score_dist.html'))
    
    # Correlation Analysis (Numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig_corr = px.imshow(corr_matrix, 
                         text_auto='.2f', 
                         aspect="auto", 
                         title='Correlation Matrix',
                         color_continuous_scale='RdBu_r')
    fig_corr.write_html(os.path.join(OUTPUT_DIR, 'correlation_matrix.html'))
    
    print(f"EDA interactive plots saved to {OUTPUT_DIR}")

def preprocess_data(df):
    print("\n--- Preprocessing ---")
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score']
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

def train_and_evaluate_models(X, y, preprocessor):
    print("\n--- Model Training & Evaluation ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2_score(y_test, y_pred)
        })
        trained_models[name] = clf
    
    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'), index=False)
    
    # Interactive Model Comparison
    fig_models = px.bar(results_df, x='Model', y='RMSE', color='R2 Score', 
                        title='Model Performance Comparison (Lower RMSE is better)',
                        text_auto='.3f')
    fig_models.write_html(os.path.join(OUTPUT_DIR, 'model_comparison.html'))
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest Model: {best_model_name}")
    
    # Save the best model
    import joblib
    model_filename = 'best_student_model.pkl'
    joblib.dump(trained_models[best_model_name], model_filename)
    print(f"Best model saved to {model_filename}")
    
    return trained_models[best_model_name], results_df, X_train

def feature_importance(model_pipeline, X_train, model_name="Best Model"):
    print(f"\n--- Feature Importance ({model_name}) ---")
    
    preprocessor = model_pipeline.named_steps['preprocessor']
    try:
        num_features = preprocessor.transformers_[0][2]
        cat_transformer = preprocessor.transformers_[1][1]
        encoder = cat_transformer.named_steps['encoder']
        cat_features_encoded = encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        feature_names = list(num_features) + list(cat_features_encoded)
        
        regressor = model_pipeline.named_steps['regressor']
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
        elif hasattr(regressor, 'coef_'):
            importances = np.abs(regressor.coef_)
        else:
            return

        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        # Interactive Feature Importance
        fig_fi = px.bar(fi_df.head(20), x='Importance', y='Feature', orientation='h',
                        title=f'Top 20 Feature Importance - {model_name}',
                        color='Importance')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        fig_fi.write_html(os.path.join(OUTPUT_DIR, 'feature_importance.html'))
        print(f"Feature importance interactive plot saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    perform_eda(df)
    X, y, preprocessor = preprocess_data(df)
    best_model, results_df, X_train = train_and_evaluate_models(X, y, preprocessor)
    feature_importance(best_model, X_train, model_name=results_df.iloc[0]['Model'])
