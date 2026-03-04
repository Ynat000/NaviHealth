from flask import Flask, request, jsonify
from flask_cors import CORS
from main import healthbc_rag  # Import your function from main.py

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/api/process', methods=['POST'])
def process_string():
    try:
        # Get the input string from the request
        data = request.get_json()
        input_string = data.get('input', '')
        
        # Call your function
        result = healthbc_rag(input_string)
        
        # Return the result
        return jsonify({'result': result}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)