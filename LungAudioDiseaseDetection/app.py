from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import numpy as np
import librosa
import tensorflow as tf
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
import plotly
import plotly.graph_objects as go
import plotly.express as px
import json
from PIL import Image
import sys

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Define the upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model classes
lung_classes = ['COVID-19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
audio_classes = ['COPD', 'Bronchiolitis', 'URTI', 'Healthy']

# Global variables for model status
LUNG_MODEL_LOADED = False
AUDIO_MODEL_LOADED = False
lungmodel = None
audiomodel = None

# Initialize models
try:
    # Load lung model
    from LungDiseaseClassification.Web_Model import Web_Model
    lungmodel = Web_Model(os.path.join("LungDiseaseClassification", "best_weight.h5"), lung_classes)
    LUNG_MODEL_LOADED = True
    print("Lung model loaded successfully")
except Exception as e:
    print(f"Error loading lung model: {e}")
    LUNG_MODEL_LOADED = False

try:
    # Load audio model
    audiomodel = load_model(os.path.join("Audio_Classification", "Lungs_GRU_CNN_1.keras"))
    AUDIO_MODEL_LOADED = True
    print("Audio model loaded successfully")
except Exception as e:
    print(f"Error loading audio model: {e}")
    AUDIO_MODEL_LOADED = False

# Define the labels and treatment information
treatment_info = {
    "COPD": """Treatment for COPD:<br><br>

    1. <b>Smoking Cessation:</b> The most crucial step. Consider smoking cessation programs and medications.<br><br>
    
    2. <b>Medications:</b><br>
       - <b>Bronchodilators:</b> Short-acting (e.g., Albuterol) and long-acting (e.g., Salmeterol) to help relax airways.<br>
       - <b>Inhaled Corticosteroids:</b> Reduces airway inflammation (e.g., Fluticasone).<br>
       - <b>Combination Inhalers:</b> Combine bronchodilators and corticosteroids (e.g., Advair).<br>
       - <b>Phosphodiesterase-4 Inhibitors:</b> Reduces inflammation and relaxes airways (e.g., Roflumilast).<br><br>

    3. <b>Oxygen Therapy:</b> For those with low blood oxygen levels.<br><br>
    
    4. <b>Pulmonary Rehabilitation:</b> Includes exercise training, nutrition advice, and education.<br><br>
    
    5. <b>Vaccinations:</b> Annual flu shots and pneumococcal vaccines.<br><br>

    <b>Lifestyle Changes:</b><br>
    - Regular exercise<br>
    - Balanced diet<br>
    - Avoiding pollutants and irritants<br><br>

    <b>Emergency Management:</b><br>
    - Have an action plan for worsening symptoms.
    """,
    "Bronchiolitis": """Treatment for Bronchiolitis:<br><br>

    1. <b>Supportive Care:</b><br>
       - <b>Hydration:</b> Ensure adequate fluid intake.<br>
       - <b>Nutrition:</b> Maintain proper nutrition and feeding.<br>
       - <b>Rest:</b> Encourage rest and reduce physical exertion.<br><br>

    2. <b>Medications:</b><br>
       - <b>Bronchodilators:</b> For relief of wheezing and shortness of breath (e.g., Albuterol).<br>
       - <b>Steroids:</b> Sometimes used in severe cases to reduce inflammation (e.g., Prednisolone).<br><br>

    3. <b>Oxygen Therapy:</b> For patients with low oxygen levels.<br><br>

    4. <b>Respiratory Therapy:</b><br>
       - <b>Nebulizers:</b> Use of nebulized medications to open airways.<br>
       - <b>Chest Physiotherapy:</b> Techniques to help clear mucus.<br><br>

    5. <b>Monitoring:</b> Regular monitoring of respiratory status and adjustments in care as needed.
    """,
    "Pneumonia": """Treatment for Pneumonia:<br><br>

    1. <b>Antibiotics:</b> Choice depends on the type of pneumonia and patient’s health status.<br>
       - <b>Macrolides:</b> Azithromycin, Clarithromycin<br>
       - <b>Penicillins:</b> Amoxicillin, Penicillin<br>
       - <b>Cephalosporins:</b> Ceftriaxone, Cefotaxime<br><br>

    2. <b>Supportive Care:</b><br>
       - <b>Hydration:</b> Maintain hydration.<br>
       - <b>Rest:</b> Ensure adequate rest.<br><br>

    3. <b>Oxygen Therapy:</b> For those with hypoxemia or severe pneumonia.<br><br>

    4. <b>Antiviral/Antifungal Medications:</b> If pneumonia is caused by a virus or fungus.<br><br>

    5. <b>Pain Management:</b> Medications to alleviate chest pain.<br><br>

    6. <b>Vaccinations:</b> Pneumococcal vaccine and influenza vaccine.
    """,
    "URTI": """Treatment for URTI:<br><br>

    1. <b>Supportive Care:</b><br>
       - <b>Hydration:</b> Drink plenty of fluids.<br>
       - <b>Rest:</b> Adequate rest to support recovery.<br><br>

    2. <b>Medications:</b><br>
       - <b>Decongestants:</b> To relieve nasal congestion (e.g., Pseudoephedrine).<br>
       - <b>Antihistamines:</b> For symptoms of allergies (e.g., Loratadine, Cetirizine).<br>
       - <b>Pain Relievers:</b> For sore throat and fever (e.g., Acetaminophen, Ibuprofen).<br><br>

    3. <b>Home Remedies:</b><br>
       - <b>Warm Saline Gargles:</b> To soothe a sore throat.<br>
       - <b>Steam Inhalation:</b> To relieve nasal congestion.<br><br>

    4. <b>Avoidance of Irritants:</b> Stay away from smoke and other irritants.
    """,
    "Healthy": """For Healthy Individuals:<br><br>

    1. <b>Healthy Lifestyle:</b><br>
       - <b>Balanced Diet:</b> Include a variety of nutrients.<br>
       - <b>Regular Exercise:</b> Engage in physical activity to maintain fitness.<br>
       - <b>Adequate Sleep:</b> Ensure quality sleep each night.<br><br>

    2. <b>Preventive Measures:</b><br>
       - <b>Vaccinations:</b> Keep up with routine vaccinations.<br>
       - <b>Regular Check-ups:</b> Schedule periodic health check-ups.
    """
}

# Disease-specific treatment and precaution information
LUNG_DISEASE_INFO = {
    'COVID-19': {
        'treatment': {
            'immediate_steps': [
                'Isolate immediately to prevent transmission',
                'Monitor oxygen saturation levels regularly',
                'Start prescribed antiviral medications if indicated'
            ],
            'medications': [
                'Antipyretics for fever (e.g., Acetaminophen)',
                'Oral hydration with electrolyte solutions',
                'Prescribed antivirals if severe'
            ],
            'long_term': [
                'Pulmonary rehabilitation exercises',
                'Regular follow-up for 3 months',
                'Gradual return to activities'
            ]
        },
        'precautions': {
            'isolation': 'Minimum 10 days isolation from symptom onset',
            'monitoring': [
                'Check oxygen levels 3 times daily',
                'Monitor temperature every 4-6 hours',
                'Track any new or worsening symptoms'
            ],
            'warning_signs': [
                'Oxygen saturation below 94%',
                'Severe difficulty breathing',
                'Persistent chest pain',
                'Confusion or inability to stay awake'
            ]
        }
    },
    'PNEUMONIA': {
        'treatment': {
            'immediate_steps': [
                'Start prescribed antibiotics immediately',
                'Rest and hydration',
                'Position patient for optimal breathing'
            ],
            'medications': [
                'Prescribed antibiotics (specific to bacterial type)',
                'Antipyretics for fever',
                'Expectorants if prescribed'
            ],
            'long_term': [
                'Complete full course of antibiotics',
                'Follow-up chest X-ray in 4-6 weeks',
                'Breathing exercises'
            ]
        },
        'precautions': {
            'lifestyle': [
                'Stop smoking if applicable',
                'Avoid exposure to pollutants',
                'Practice good oral hygiene'
            ],
            'monitoring': [
                'Track temperature and breathing rate',
                'Monitor for worsening symptoms',
                'Keep track of medication schedule'
            ],
            'warning_signs': [
                'High fever persisting > 3 days on antibiotics',
                'Worsening shortness of breath',
                'Coughing up blood'
            ]
        }
    },
    'TUBERCULOSIS': {
        'treatment': {
            'immediate_steps': [
                'Begin DOTS therapy immediately',
                'Contact tracing of close contacts',
                'Isolation until sputum conversion'
            ],
            'medications': [
                'Isoniazid (INH)',
                'Rifampicin (RIF)',
                'Pyrazinamide (PZA)',
                'Ethambutol (EMB)'
            ],
            'long_term': [
                'Complete 6-month treatment regimen',
                'Regular liver function monitoring',
                'Monthly sputum tests',
                'Nutritional support'
            ]
        },
        'precautions': {
            'infection_control': [
                'Wear N95 mask in public',
                'Ensure good ventilation',
                'Cover mouth while coughing'
            ],
            'monitoring': [
                'Monthly weight check',
                'Regular sputum tests',
                'Monitor for medication side effects'
            ],
            'warning_signs': [
                'Hemoptysis',
                'Severe weight loss',
                'Yellow skin or eyes (jaundice)',
                'Severe nausea or abdominal pain'
            ]
        }
    },
    'NORMAL': {
        'treatment': {
            'preventive_care': [
                'Regular health check-ups',
                'Maintain good physical activity',
                'Balanced nutrition'
            ],
            'lifestyle': [
                'Regular exercise',
                'Adequate sleep',
                'Stress management'
            ]
        },
        'precautions': {
            'general': [
                'Practice good hand hygiene',
                'Stay up to date with vaccinations',
                'Maintain a healthy lifestyle'
            ],
            'monitoring': [
                'Annual health check-ups',
                'Regular exercise routine',
                'Balanced diet'
            ]
        }
    }
}

RESPIRATORY_CONDITION_INFO = {
    'COPD': {
        'treatment': {
            'immediate_steps': [
                'Assess severity using spirometry',
                'Start bronchodilator therapy',
                'Evaluate oxygen needs'
            ],
            'medications': [
                'Short-acting bronchodilators (SABA)',
                'Long-acting bronchodilators (LABA)',
                'Inhaled corticosteroids if indicated'
            ],
            'long_term': [
                'Pulmonary rehabilitation program',
                'Smoking cessation support',
                'Regular follow-up every 3-6 months'
            ]
        },
        'precautions': {
            'lifestyle': [
                'Quit smoking immediately',
                'Avoid air pollutants',
                'Regular exercise as tolerated'
            ],
            'monitoring': [
                'Daily symptom diary',
                'Peak flow measurements',
                'Oxygen saturation monitoring'
            ],
            'warning_signs': [
                'Increased breathlessness',
                'Change in sputum color',
                'Decreased exercise tolerance'
            ]
        }
    },
    'Bronchiolitis': {
        'treatment': {
            'immediate_steps': [
                'Assess respiratory status',
                'Ensure adequate hydration',
                'Nasal suctioning if needed'
            ],
            'supportive_care': [
                'Humidified oxygen if needed',
                'Small frequent feeds',
                'Position upright for feeding'
            ],
            'monitoring': [
                'Respiratory rate and effort',
                'Feeding tolerance',
                'Hydration status'
            ]
        },
        'precautions': {
            'home_care': [
                'Regular nasal suctioning',
                'Humidified air',
                'Smaller, frequent feeds'
            ],
            'monitoring': [
                'Breathing rate and effort',
                'Wet diapers/hydration',
                'Temperature'
            ],
            'warning_signs': [
                'Increased work of breathing',
                'Poor feeding',
                'Lethargy or irritability'
            ]
        }
    },
    'URTI': {
        'treatment': {
            'immediate_steps': [
                'Rest and hydration',
                'Symptomatic relief',
                'Monitor for complications'
            ],
            'medications': [
                'Antipyretics if needed',
                'Saline nasal drops',
                'Throat lozenges if indicated'
            ],
            'supportive_care': [
                'Adequate rest',
                'Increased fluid intake',
                'Humidification'
            ]
        },
        'precautions': {
            'lifestyle': [
                'Rest voice if laryngitis',
                'Avoid irritants',
                'Good hand hygiene'
            ],
            'monitoring': [
                'Temperature',
                'Hydration status',
                'Breathing comfort'
            ],
            'warning_signs': [
                'High fever > 3 days',
                'Difficulty breathing',
                'Severe throat pain'
            ]
        }
    },
    'Healthy': {
        'recommendations': {
            'maintenance': [
                'Regular exercise',
                'Balanced diet',
                'Adequate sleep'
            ],
            'prevention': [
                'Regular health check-ups',
                'Good respiratory hygiene',
                'Avoid smoking and pollutants'
            ]
        },
        'lifestyle': {
            'exercise': [
                'Regular aerobic activity',
                'Deep breathing exercises',
                'Stress management'
            ],
            'diet': [
                'Balanced nutrition',
                'Adequate hydration',
                'Antioxidant-rich foods'
            ]
        }
    }
}

def LungDiseaseDetection(file_path):
    """
    Detects lung disease from an image file using a pre-trained model.

    Args:
        file_path (str): Path to the image file to process.

    Returns:
        dict: Prediction result with class and confidence.
    """
    try:
        # Open and preprocess the image
        image = Image.open(file_path).convert("RGB")
        
        # Make prediction using the Web_Model class
        predicted_class = lungmodel.predict(image)
        
        # Get confidence (mock value since Web_Model doesn't return it)
        confidence = 95.0  # Using a default high confidence
        
        print(f"Predicted class: {predicted_class}, Confidence: {confidence}%")
        
        return {
            'class': predicted_class,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Error processing the image: {e}")
        return None

def preprocess_audio(file_path):
    """Preprocesses the audio file to extract features."""
    try:
        # Load the audio file
        data_x, sampling_rate = librosa.load(file_path, duration=None, sr=22050)
        data_x = np.array(data_x, copy=True)
        
        # Ensure minimum length
        if len(data_x) < sampling_rate * 2:
            data_x = np.pad(data_x, (0, sampling_rate * 2 - len(data_x)), mode='constant')
        
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=data_x, sr=sampling_rate, n_mfcc=52)
        
        if mfcc_features.shape[1] == 0:
            print("Error: No MFCC features extracted")
            return None
        
        # Calculate mean of features
        features = np.mean(mfcc_features.T, axis=0)
        
        # Reshape to match model's expected input shape (None, 1, 52)
        features = np.array(features, copy=True).reshape(1, 1, 52)
        
        print(f"Final feature shape: {features.shape}")
        return features
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def stretch(data, stretch_factor):
    """Stretches or compresses the audio signal."""
    try:
        input_length = len(data)
        output_length = int(input_length * stretch_factor)
        output_data = np.zeros(output_length)
        
        for i in range(output_length):
            input_index = int(i / stretch_factor)
            if input_index < input_length:
                output_data[i] = data[input_index]
        
        return output_data
    except Exception as e:
        print(f"Error stretching audio: {e}")
        return data

def generate_report_data(prediction, patient_info, diagnosis_type):
    """Generate report data based on prediction results."""
    report_data = {
        'patient': {
            'name': patient_info.get('name', 'N/A'),
            'id': patient_info.get('id', 'N/A'),
            'age': patient_info.get('age', 'N/A'),
            'gender': patient_info.get('gender', 'N/A'),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'diagnosis': {
            'type': diagnosis_type,
            'result': prediction,
            'confidence': prediction.get('confidence', 0) if isinstance(prediction, dict) else 0,
            'severity': 'Moderate',  # You can implement severity calculation logic
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'charts': {}
    }

    # Generate confidence gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=report_data['diagnosis']['confidence'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    report_data['charts']['confidence_gauge'] = json.loads(fig.to_json())

    # Add treatment information
    if isinstance(prediction, dict) and 'class' in prediction:
        report_data['treatment'] = treatment_info.get(prediction['class'], '')
    elif isinstance(prediction, str):
        report_data['treatment'] = treatment_info.get(prediction, '')

    return report_data

# Routes
@app.route('/')
def main_home():
    return render_template('index.html')

@app.route('/lung-diagnosis', methods=['GET'])
def index():
    return render_template('lung_diagnosis.html')

@app.route('/predict-lung', methods=['POST'])
def predict_lung():
    """
    Handles image upload and prediction, returning the result as JSON.
    """
    if not LUNG_MODEL_LOADED or lungmodel is None:
        return jsonify({'error': 'Lung model not loaded. Please ensure the model is properly initialized.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Processing lung image: {filename}")

        # Make prediction
        result = LungDiseaseDetection(filepath)
        
        if result is None:
            return jsonify({'error': 'Error processing the image. Please ensure it is a valid image file.'}), 500

        # Get patient info from form data
        patient_info = {
            'name': request.form.get('patient_name', 'N/A'),
            'id': request.form.get('patient_id', 'N/A'),
            'age': request.form.get('age', 'N/A'),
            'gender': request.form.get('gender', 'N/A')
        }

        # Generate report data
        report_data = generate_report_data(result, patient_info, 'Lung Analysis')
        
        # Return prediction data
        response = {
            'prediction': result,
            'patient_info': patient_info,
            'report_data': report_data
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {e}")

@app.route('/audio-diagnosis')
def audio_home():
    return render_template('audio_diagnosis.html')

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if not AUDIO_MODEL_LOADED or audiomodel is None:
        return jsonify({'error': 'Audio model not loaded. Please ensure the model is properly initialized.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Processing audio file: {filename}")

        # Preprocess the audio
        features = preprocess_audio(filepath)
        if features is None:
            return jsonify({'error': 'Error processing audio file. Please ensure it is a valid audio file.'}), 500

        print(f"Feature shape before prediction: {features.shape}")

        # Make prediction
        prediction = audiomodel.predict(features)
        print(f"Raw prediction shape: {prediction.shape}")
        print(f"Raw prediction values: {prediction}")
        
        predicted_class = audio_classes[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]) * 100)

        print(f"Predicted class: {predicted_class}, Confidence: {confidence}%")

        result = {
            'class': predicted_class,
            'confidence': confidence
        }

        # Get patient info from form data
        patient_info = {
            'name': request.form.get('patient_name', 'N/A'),
            'id': request.form.get('patient_id', 'N/A'),
            'age': request.form.get('age', 'N/A'),
            'gender': request.form.get('gender', 'N/A')
        }

        # Return prediction data
        response = {
            'prediction': result,
            'patient_info': patient_info
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Error during prediction: {str(e)}"}), 500
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error removing temporary file: {e}")

@app.route('/generate_audio_report', methods=['POST'])
def generate_audio_report():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        prediction = data.get('prediction')
        patient_info = data.get('patient_info')

        if not prediction or not patient_info:
            return jsonify({'error': 'Missing prediction or patient information'}), 400

        # Get condition-specific information
        condition = prediction.get('class', 'Healthy')
        condition_info = RESPIRATORY_CONDITION_INFO.get(condition, RESPIRATORY_CONDITION_INFO['Healthy'])

        # Generate report data
        report_data = {
            'prediction': prediction,
            'patient_info': patient_info,
            'current_date': datetime.now().strftime('%B %d, %Y'),
            'report_id': f"AR{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'condition_info': condition_info
        }

        # Render the report template
        return render_template('audio_report.html', **report_data)

    except Exception as e:
        print(f"Error generating audio report: {e}")
        return jsonify({'error': 'Error generating report'}), 500

@app.route('/generate_lung_report', methods=['POST'])
def generate_lung_report():
    """Generate a detailed report for lung diagnosis."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        prediction = data.get('prediction')
        patient_info = data.get('patient_info')

        if not prediction or not patient_info:
            return jsonify({'error': 'Missing prediction or patient information'}), 400

        # Generate report data
        report_data = generate_report_data(prediction, patient_info, 'Lung Analysis')

        # Add lung-specific information
        report_data.update({
            'diagnosis_type': 'Lung X-Ray Analysis',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'disease_info': LUNG_DISEASE_INFO.get(prediction['class'], {}),
            'severity': 'High' if prediction['confidence'] > 90 else 'Moderate',
            'recommendations': [
                'Follow-up chest X-ray in 2-4 weeks',
                'Complete prescribed medication course',
                'Monitor symptoms and report any changes',
                'Maintain good respiratory hygiene'
            ]
        })

        # Render the report template
        return render_template(
            'lung_report.html',
            report_data=report_data,
            patient_info=patient_info,
            prediction=prediction
        )

    except Exception as e:
        print(f"Error generating lung report: {str(e)}")
        return jsonify({'error': f"Error generating report: {str(e)}"}), 500

# Only run the app if this file is run directly
if __name__ == '__main__':
    app.run(debug=True, port=5000)