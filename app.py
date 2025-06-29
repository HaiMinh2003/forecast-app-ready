from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet

app = Flask(__name__)
CORS(app)

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

    # Tính phần nhận xét dựa trên trung bình dự báo
    forecasted_mean = forecast['yhat'].tail(forecast_months).mean()
    pct_total_change = (forecasted_mean - recent_avg) / recent_avg * 100

    if pct_total_change > 10:
        comment = f"Biểu đồ cho thấy xu hướng TĂNG, doanh thu dự kiến tăng khoảng {pct_total_change:.1f}% so với trung bình 3 tháng gần nhất."
    elif pct_total_change < -10:
        comment = f"Biểu đồ cho thấy xu hướng GIẢM, doanh thu dự kiến giảm khoảng {abs(pct_total_change):.1f}% so với trung bình 3 tháng gần nhất."
    else:
        comment = "Biểu đồ cho thấy doanh thu dự báo ỔN ĐỊNH, không biến động lớn."

    result = forecast[['ds', 'yhat', 'delta', 'pct_change']].tail(forecast_months).to_dict(orient='records')

    # Trả thêm comment trong JSON
    return jsonify({
        'forecast': result,
        'comment': comment
    })

if __name__ == '__main__':
    app.run(debug=True)
