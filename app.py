from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('catboost_model.pkl')
encoders = joblib.load('encoders_dict.pkl')

field_map = {
    "gender": "Gender",
    "age": "Age",
    "education_level": "Education Level",
    "institute_type": "Institution Type",
    "it_student": "IT Student",
    "location": "Location",
    "load_shedding": "Load-shedding",
    "financial_condition": "Financial Condition",
    "internet_type": "Internet Type",
    "network_type": "Network Type",
    "class_duration": "Class Duration",
    "self_lms": "Self Lms",
    "device": "Device"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def index():
    return render_template('index.html')

@app.route('/adapt')
def adapt():
    return render_template('adaptivity.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        
        input_data = {}
        for key, value in request.form.items():
            mapped_key = field_map.get(key, key)  
            input_data[mapped_key] = value

        input_df = pd.DataFrame([input_data])

        
        for col in input_df.columns:
            if col in encoders:
                le = encoders[col]
                if input_df[col].values[0] not in le.classes_:
                    return f"Invalid input for '{col}': '{input_df[col].values[0]}'"
                input_df[col] = le.transform(input_df[col])

        
        if 'Adaptivity Level' in input_df.columns:
            input_df.drop(columns=['Adaptivity Level'], inplace=True)

        
        prediction = model.predict(input_df)[0]
        adaptivity_level = 'High' if prediction == 0 else ('Moderate' if prediction == 2 else 'Low')
        return render_template("results.html", prediction=adaptivity_level)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)