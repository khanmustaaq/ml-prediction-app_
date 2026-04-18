from flask import Flask, render_template, request
import joblib
import os
import io
import boto3
from datetime import datetime

app = Flask(__name__)

# ── S3 config ────────────────────────────────────────────────────────────────
S3_BUCKET = os.environ.get('S3_BUCKET', 'ml-models-ayesha')      
S3_PREFIX = os.environ.get('S3_PREFIX', 'models')
BASE      = os.path.dirname(os.path.abspath(__file__))


def load_model(model_name: str):
    """Load from S3 if bucket configured, else fall back to local models/ folder."""
    if S3_BUCKET:
        s3 = boto3.client('s3')
        key = f"{S3_PREFIX}/{model_name}"
        print(f"[S3] Loading s3://{S3_BUCKET}/{key}")
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, key, buf)
        buf.seek(0)
        return joblib.load(buf)
    else:
        local_path = os.path.join(BASE, 'models', model_name)
        print(f"[LOCAL] Loading {local_path}")
        return joblib.load(local_path)


# ── Load all three models at startup ─────────────────────────────────────────
telecom_model    = load_model('telecom_model.pkl')
healthcare_model = load_model('healthcare_model.pkl')
ecommerce_model  = load_model('ecommerce_model.pkl')


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/telecom', methods=['GET', 'POST'])
def telecom():
    result = None
    if request.method == 'POST':
        gender           = int(request.form['gender'])
        partner          = int(request.form['partner'])
        dependents       = int(request.form['dependents'])
        tenure           = float(request.form['tenure'])
        online_security  = int(request.form['online_security'])
        online_backup    = int(request.form['online_backup'])
        device_protection= int(request.form['device_protection'])
        tech_support     = int(request.form['tech_support'])
        contract         = int(request.form['contract'])
        monthly_charges  = float(request.form['monthly_charges'])

        prediction = telecom_model.predict([[
            gender, partner, dependents, tenure,
            online_security, online_backup, device_protection,
            tech_support, contract, monthly_charges
        ]])
        result = 'Customer is likely to CHURN' if prediction[0] == 1 else 'Customer is NOT likely to churn'

    return render_template('telecom.html', result=result)


@app.route('/healthcare', methods=['GET', 'POST'])
def healthcare():
    result = None
    if request.method == 'POST':
        age            = float(request.form['age'])
        gender         = int(request.form['gender'])
        diagnosis      = int(request.form['diagnosis'])
        admission_date = request.form['admission_date']
        discharge_date = request.form['discharge_date']
        treatment_type = int(request.form['treatment_type'])
        hospital_cost  = float(request.form['hospital_cost'])

        base_date   = datetime(2025, 1, 1)
        adm_encoded = (datetime.strptime(admission_date, '%Y-%m-%d') - base_date).days
        dis_encoded = (datetime.strptime(discharge_date, '%Y-%m-%d') - base_date).days

        prediction = healthcare_model.predict([[
            age, gender, diagnosis, adm_encoded, dis_encoded,
            treatment_type, hospital_cost
        ]])
        outcomes = {0: 'Critical', 1: 'Improved', 2: 'Recovered', 3: 'Stable'}
        result = f"Predicted Outcome: {outcomes[prediction[0]]}"

    return render_template('healthcare.html', result=result)


@app.route('/ecommerce', methods=['GET', 'POST'])
def ecommerce():
    result = None
    if request.method == 'POST':
        product_category = int(request.form['product_category'])
        quantity         = float(request.form['quantity'])
        unit_price       = float(request.form['unit_price'])
        total_amount     = float(request.form['total_amount'])
        payment_method   = int(request.form['payment_method'])

        prediction = ecommerce_model.predict([[
            product_category, quantity, unit_price, total_amount, payment_method
        ]])
        statuses = {0: 'Cancelled', 1: 'Delivered', 2: 'Pending', 3: 'Returned'}
        result = f"Order Status: {statuses[prediction[0]]}"

    return render_template('ecommerce.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
