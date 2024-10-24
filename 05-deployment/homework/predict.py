import pickle
from flask import Flask, request, jsonify

# Load model files
dv = None
model = None

dv_input_file = 'dv.bin'
with open(dv_input_file, 'rb') as f_in:
  dv = pickle.load(f_in)

model_input_file = 'model2.bin'
with open(model_input_file, 'rb') as f_in:
  model = pickle.load(f_in)


# Create prediction web service
app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
  customer = request.get_json()

  X = dv.transform([customer])
  y_pred = model.predict_proba(X)[0, 1]
  subscription = y_pred >= .5

  response = {
    'subscription_probability': float(y_pred),
    'subscription': bool(subscription)
  }

  return jsonify(response)

if __name__=="__main__":
  app.run(debug=True, host='0.0.0.0', port=9696)