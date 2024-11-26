from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load model files
dv = None
model = None

input_file = "model_decisiontree.bin"
with open(input_file, 'rb') as f_in:
  dv, model = pickle.load(f_in)

# Create prediction web service
app = Flask('success')

@app.route('/predict', methods=['POST'])
def predict():
  climber = request.get_json()

  X = pd.DataFrame(
    data=dv.transform([climber]),
    columns=dv.feature_names_
  )
  
  y_pred = model.predict_proba(X)[0, 1]
  success = y_pred >= .5

  response = {
    'success_probability': float(y_pred),
    'success': bool(success)
  }

  return jsonify(response)

if __name__=="__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)