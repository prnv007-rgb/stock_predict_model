import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

plt.style.use("fivethirtyeight")

app = Flask(__name__, static_folder="static")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load your trained model
model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock') or 'POWERGRID.NS'
        
        # Download stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)
        df = yf.download(stock, start=start, end=end)

        if df.empty:
            return render_template('index.html', error="Invalid ticker or no data available.")

        # Descriptive statistics
        data_desc = df.describe()

        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Train/test split
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_training)
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        # Prepare test data
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Save plots
        ema_20_50_path = "static/ema_20_50.png"
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.Close, 'y', label='Closing Price')
        ax1.plot(ema20, 'g', label='EMA 20')
        ax1.plot(ema50, 'r', label='EMA 50')
        ax1.legend(); ax1.set_title("20 & 50 Days EMA")
        fig1.savefig(ema_20_50_path); plt.close(fig1)

        ema_100_200_path = "static/ema_100_200.png"
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.Close, 'y', label='Closing Price')
        ax2.plot(ema100, 'g', label='EMA 100')
        ax2.plot(ema200, 'r', label='EMA 200')
        ax2.legend(); ax2.set_title("100 & 200 Days EMA")
        fig2.savefig(ema_100_200_path); plt.close(fig2)

        prediction_path = "static/stock_prediction.png"
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(y_test, 'g', label="Original")
        ax3.plot(y_predicted, 'r', label="Predicted")
        ax3.legend(); ax3.set_title("Prediction vs Original")
        fig3.savefig(prediction_path); plt.close(fig3)

        # Save CSV dataset
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render_template(
            'index.html',
            plot_path_ema_20_50="ema_20_50.png",
            plot_path_ema_100_200="ema_100_200.png",
            plot_path_prediction="stock_prediction.png",
            data_desc=data_desc.to_html(classes='table table-bordered'),
            dataset_link=csv_file_path
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
