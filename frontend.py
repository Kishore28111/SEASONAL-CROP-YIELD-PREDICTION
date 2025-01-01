from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve and preprocess form data
            Year = int(request.form['Year'])  # Convert to integer
            average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
            pesticides_tonnes = float(request.form['pesticides_tonnes'])
            avg_temp = float(request.form['avg_temp'])
            Area = request.form['Area']  # Assuming preprocessor can handle strings
            Item = request.form['Item']  # Assuming preprocessor can handle strings
            
            # Create a feature array for prediction
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
            
            # Transform the features using the preprocessor
            transformed_features = preprocessor.transform(features)
            
            # Make predictions
            predicted_value = dtr.predict(transformed_features)[0]
            
            # Return the prediction result
            return render_template('index.html', predicted_value=predicted_value)
        
        except KeyError as e:
            # Handle missing form fields
            return f"Missing input field: {e}", 400
        except Exception as e:
            # General error handling
            return f"An error occurred: {e}", 500

# Main entry point for the app
if __name__ == '__main__':
    app.run(debug=True)