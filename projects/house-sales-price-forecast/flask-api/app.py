from flask import Flask, request, jsonify
import pandas as pd
from models.model_handler import ModelHandler
from config import Config
import logging

# Configurar logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar o Flask
app = Flask(__name__)

# Inicializar o manipulador do modelo
model_handler = ModelHandler(model_path=Config.MODEL_PATH)

# Logs antes e depois de cada requisição
@app.before_request
def log_request_info():
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Request data: {request.get_json()}")

@app.after_request
def log_response_info(response):
    logger.info(f"Response status: {response.status_code}")
    return response

# Rota para predições
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            logger.warning("No input data provided.")
            return jsonify({"error": "No input data provided"}), 400
        
        df = pd.DataFrame(input_data)

        predictions = model_handler.predict(df= df)
        logger.info("Prediction completed successfully.")
        return jsonify({'predictions': predictions})
    except ValueError as ve:
        logger.error(f"Value error during prediction: {str(ve)}")
        return jsonify({"error": "Value Error", "message": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)