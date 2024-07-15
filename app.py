from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'regression_data.csv')
model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'regression_model.pkl')
graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'regression_plot.png')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def save_data(data):
    """Save new data to the existing data file."""
    if os.path.exists(data_file_path):
        existing_data = pd.read_csv(data_file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        updated_data = data
    updated_data.to_csv(data_file_path, index=False)


def train_model():
    """Train the Linear Regression model using the data from the data file."""
    data = pd.read_csv(data_file_path)
    X = data[['size']]
    y = data['price']

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_path)

    # Generate and save the graph with regression line
    plt.figure()
    plt.scatter(data['size'], data['price'], color='blue')
    plt.plot(data['size'], model.predict(X), color='red')
    plt.xlabel('Size')
    plt.ylabel('Price')
    plt.title('Linear Regression')
    plt.savefig(graph_path)
    plt.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)

        if 'size' in data.columns and 'price' in data.columns:
            save_data(data)
            train_model()
            return jsonify(
                {'message': 'Data uploaded and model trained successfully!', 'graph_url': url_for('get_graph')})
        else:
            return jsonify({'error': 'CSV file must contain size and price columns'}), 400


@app.route('/graph')
def get_graph():
    return send_file(graph_path, mimetype='image/png')


@app.route('/predict', methods=['POST'])
def predict():
    size = request.form['size']
    data = [[float(size)]]

    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found, please upload a CSV file to train the model'}), 400

    model = joblib.load(model_path)
    prediction = model.predict(data)
    return jsonify({'prediction': float(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
