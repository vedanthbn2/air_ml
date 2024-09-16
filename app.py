import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Load dataset for visualization
df = pd.read_csv('city_day.csv')
df.columns = [col.strip() for col in df.columns]  # Clean column names
df['AQI'] = df[['SO2', 'NO2', 'RSPM/PM10', 'SPM']].mean(axis=1)

# Home page
@app.route('/')
def home():
    so2 = request.args.get('so2', '')
    no2 = request.args.get('no2', '')
    pm10 = request.args.get('pm10', '')
    spm = request.args.get('spm', '')
    return render_template('index.html', so2=so2, no2=no2, pm10=pm10, spm=spm)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        so2 = float(request.form['so2'])
        no2 = float(request.form['no2'])
        pm10 = float(request.form['pm10'])
        spm = float(request.form['spm'])

        features = np.array([[so2, no2, pm10, spm]])
        prediction = model.predict(features)[0]
        return redirect(url_for('result', prediction=prediction, so2=so2, no2=no2, pm10=pm10, spm=spm))
    except Exception as e:
        print(f"Error: {e}")
        return redirect(url_for('result', prediction='Error during prediction'))

# Visualization route
@app.route('/visualize')
def visualize():
    try:
        # Create bar graph
        plt.figure(figsize=(10, 6))
        sns.barplot(x='City/Town/Village/Area', y='SO2', data=df[:10])  # Example: SO2 for the first 10 cities
        plt.title('SO2 Levels in Different Cities')
        plt.xticks(rotation=45)
        plt.tight_layout()
        bar_chart_path = 'static/bar_chart.png'
        plt.savefig(bar_chart_path)
        plt.close()

        # Create pie chart for AQI categories (Assuming you have a column for AQI categories)
        if 'AQI_Category' in df.columns:
            plt.figure(figsize=(6, 6))
            aqi_counts = df['AQI_Category'].value_counts()
            plt.pie(aqi_counts, labels=aqi_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ffcc99','#ff9999','#ffb3e6'])
            plt.title('AQI Category Distribution')
            pie_chart_path = 'static/pie_chart.png'
            plt.savefig(pie_chart_path)
            plt.close()
        else:
            pie_chart_path = None

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='SO2', y='NO2', data=df)
        plt.title('SO2 vs NO2')
        scatter_plot_path = 'static/scatter_plot.png'
        plt.savefig(scatter_plot_path)
        plt.close()

        # Create histogram of SO2
        plt.figure(figsize=(8, 6))
        sns.histplot(df['SO2'], bins=30, kde=True) # type: ignore
        plt.title('Distribution of SO2')
        histogram_path = 'static/histogram.png'
        plt.savefig(histogram_path)
        plt.close()

        # Create box plot for NO2
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='NO2', data=df)
        plt.title('Box Plot of NO2')
        box_plot_path = 'static/box_plot.png'
        plt.savefig(box_plot_path)
        plt.close()

        # Create pairwise plot
        plt.figure(figsize=(10, 8))
        sns.pairplot(df[['SO2', 'NO2', 'RSPM/PM10', 'SPM']])
        plt.title('Pairwise Plot')
        pairplot_path = 'static/pairplot.png'
        plt.savefig(pairplot_path)
        plt.close()

        # Render the visualization page with generated images
        return render_template('visualization.html', bar_chart=bar_chart_path, pie_chart=pie_chart_path, scatter_plot=scatter_plot_path, histogram=histogram_path, box_plot=box_plot_path, pairplot=pairplot_path)

    except Exception as e:
        print(f"Visualization Error: {e}")
        return redirect(url_for('home'))

# Result route
@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'Error during prediction')
    so2 = request.args.get('so2', '')
    no2 = request.args.get('no2', '')
    pm10 = request.args.get('pm10', '')
    spm = request.args.get('spm', '')
    return render_template('result.html', prediction=prediction, so2=so2, no2=no2, pm10=pm10, spm=spm)

# Clear form route
@app.route('/clear')
def clear():
    return render_template('index.html', so2='', no2='', pm10='', spm='')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
