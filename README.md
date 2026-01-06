# Healthcare Cost Prediction & Explanation

This application predicts healthcare costs based on patient data and provides explanations for the predictions using machine learning and explainable AI techniques.

## Features

- Predict annual healthcare costs based on patient demographics and health data
- Visualize key factors influencing the prediction
- Get AI-generated explanations of cost factors
- Upload CSV files for batch predictions
- Interactive form for single patient predictions

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone this repository or download the source code
2. Navigate to the project directory
3. Activate the virtual environment:

```bash
# On Windows
activate_venv.bat

# On macOS/Linux
source venv/bin/activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. (Optional) Create a `.env` file in the project root and add your OpenRouter API key for LLM explanations:

```
OPENROUTER_API_KEY=your_api_key_here
```

### Running the Application

```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `models/`: Directory containing trained ML models
  - `xgboost_reg.pkl`: XGBoost regression model
  - `random_forest_reg.pkl`: Random Forest regression model
  - `reg_preprocessor.pkl`: Data preprocessor
- `data/`: Directory containing datasets
- `notebooks/`: Jupyter notebooks for model development
- `requirements.txt`: Required Python packages

## Model Information

The application uses two regression models:
1. XGBoost Regressor (primary)
2. Random Forest Regressor (fallback)

The models were trained on healthcare cost data with features including age, BMI, smoking status, and healthcare utilization metrics.

## License

This project is for educational purposes only.
