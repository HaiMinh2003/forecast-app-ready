from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet

app = Flask(__name__)
CORS(app)  # Cho phép mọi origin gọi đến API

@app.route('/forecast', methods=['POST'])
def forecast():
    file = request.files['file']
    stock_code = request.form['stockCode']
    country = request.form['country']
    forecast_months = int(request.form['months'])

    df = pd.read_csv(file, encoding='ISO-8859-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df.dropna(subset=['InvoiceDate', 'StockCode', 'Quantity', 'UnitPrice'], inplace=True)
    df['Month'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    df = df[(df['StockCode'] == stock_code) & (df['Country'] == country)]

    if df.empty:
        return jsonify({'error': 'Không có dữ liệu phù hợp'}), 400

    monthly = df.groupby('Month').agg({'Revenue': 'sum'}).reset_index()
    monthly.columns = ['ds', 'y']

    model = Prophet()
    model.fit(monthly)

    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)

    recent_avg = monthly['y'].tail(3).mean()
    forecast['delta'] = forecast['yhat'] - recent_avg
    forecast['pct_change'] = 100 * forecast['delta'] / recent_avg

    result = forecast[['ds', 'yhat', 'delta', 'pct_change']].tail(forecast_months).to_dict(orient='records')
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)