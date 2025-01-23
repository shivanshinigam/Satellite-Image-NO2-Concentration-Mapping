#!/usr/bin/env python3
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def color_distance(c1, c2):
    return np.sqrt(np.sum((np.array(c1) - np.array(c2))**2))

def rgb_to_no2(rgb):
    color_to_no2 = {
        (255, 0, 0): 100,
        (0, 255, 0): 50,
        (0, 0, 255): 10,
        (255, 255, 0): 75,
    }
    closest_color = min(color_to_no2.keys(), key=lambda color: color_distance(color, rgb))
    return color_to_no2[closest_color]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        image = Image.open(file_path)
        image_array = np.array(image)
        
        no2_values = []
        for y in range(image_array.shape[0]):
            for x in range(image_array.shape[1]):
                rgb = tuple(image_array[y, x, :3])
                no2_value = rgb_to_no2(rgb)
                no2_values.append(no2_value)
        
        average_no2 = np.mean(no2_values)
        
        draw = ImageDraw.Draw(image)
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, size=24)
        text = f"Avg NO₂: {average_no2:.2f} µg/m³"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = (image.width - text_width - 10, image.height - text_height - 10)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))
        
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
        image.save(result_image_path)
        
        place_name = request.form.get('place_name', 'Unknown')
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        data = {
            "Place Name": [place_name],
            "Date": [current_date],
            "Average NO2 Concentration (µg/m³)": [average_no2]
        }
        
        df = pd.DataFrame(data)
        file_name = "no2_concentration_data.xlsx"
        
        try:
            existing_df = pd.read_excel(file_name)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_excel(file_name, index=False)
        
        return render_template('result.html', image_url=url_for('uploaded_file', filename='result_' + filename), average_no2=average_no2, place_name=place_name, current_date=current_date)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
