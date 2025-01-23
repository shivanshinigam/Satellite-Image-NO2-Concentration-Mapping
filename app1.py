from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import random
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'xlsx', 'jpeg', 'jpg', 'png', 'bmp', 'tiff'}

# Ensure upload and static folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Constants
WHITE_THRESHOLD = 200
VARIATION_RANGE = 10
BOX_SIZE = 100
RED_THRESHOLD = 150
N_CLUSTERS = 5
BORDER_SIZE = 2

no2_to_color = {
    100: (255, 0, 0),
    50: (0, 255, 0),
    10: (0, 0, 255),
    75: (255, 255, 0),
}

def is_near_white(pixel, threshold=WHITE_THRESHOLD):
    return np.mean(pixel[:3]) > threshold

def is_red(pixel, threshold=RED_THRESHOLD):
    r, g, b = pixel[:3]
    return r > threshold and g < threshold and b < threshold

def no2_to_rgb(no2_concentration):
    closest_no2 = min(no2_to_color.keys(), key=lambda no2: abs(no2 - no2_concentration))
    return no2_to_color[closest_no2]

def get_random_shade(base_color, variation_range):
    return tuple(
        min(max(int(c + random.uniform(-variation_range, variation_range)), 0), 255)
        for c in base_color
    )

def cluster_colors(image_array, n_clusters=N_CLUSTERS):
    pixels = image_array.reshape(-1, 4)[:, :3]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_.astype(int), kmeans.labels_.reshape(image_array.shape[:2])

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'image' not in request.files or 'excel' not in request.files:
        return redirect(request.url)

    image_file = request.files['image']
    excel_file = request.files['excel']
    chosen_entry = request.form.get('chosen_entry')

    if image_file and allowed_file(image_file.filename) and excel_file and allowed_file(excel_file.filename):
        image_filename = secure_filename(image_file.filename)
        excel_filename = secure_filename(excel_file.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)

        image_file.save(image_path)
        excel_file.save(excel_path)

        return redirect(url_for('process_files', image_filename=image_filename, excel_filename=excel_filename, chosen_entry=chosen_entry))
    return redirect(url_for('index'))

@app.route('/process')
def process_files():
    image_filename = request.args.get('image_filename')
    excel_filename = request.args.get('excel_filename')
    chosen_entry = request.args.get('chosen_entry')

    if not image_filename or not excel_filename or not chosen_entry:
        return 'No entry specified'

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)

    df = pd.read_excel(excel_path)
    filtered_df = df[df[df.columns[0]] == chosen_entry]

    if filtered_df.empty:
        return 'No data available for the selected entry'

    if len(filtered_df.columns) < 3:
        return 'Error: The Excel file does not have a third column.'

    value_from_third_column = filtered_df.iloc[0, 2]
    base_color = no2_to_rgb(value_from_third_column)

    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)

    cluster_colors_list, labels = cluster_colors(image_array)

    overlay_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay_image)

    red_mask = np.apply_along_axis(is_red, 2, image_array)
    white_mask = np.apply_along_axis(lambda x: is_near_white(x), 2, image_array)
    dilated_white_mask = binary_dilation(white_mask, structure=np.ones((BOX_SIZE, BOX_SIZE)))

    for y in range(0, image.height, BOX_SIZE):
        for x in range(0, image.width, BOX_SIZE):
            region = dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]
            if np.any(region & red_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
                red_block_color = get_random_shade((255, 0, 0), VARIATION_RANGE)
                border_rect = [x, y, x + BOX_SIZE, y + BOX_SIZE]
                overlay_draw.rectangle(border_rect, outline=(0, 0, 0), width=BORDER_SIZE)
                fill_rect = [x + BORDER_SIZE, y + BORDER_SIZE, x + BOX_SIZE - BORDER_SIZE, y + BOX_SIZE - BORDER_SIZE]
                overlay_draw.rectangle(fill_rect, fill=(*red_block_color, 128))

    for y in range(0, image.height, BOX_SIZE):
        for x in range(0, image.width, BOX_SIZE):
            if np.all(dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
                if not np.any(dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE] & red_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
                    border_rect = [x, y, x + BOX_SIZE, y + BOX_SIZE]
                    overlay_draw.rectangle(border_rect, outline=(0, 0, 0), width=BORDER_SIZE)
                    fill_rect = [x + BORDER_SIZE, y + BORDER_SIZE, x + BOX_SIZE - BORDER_SIZE, y + BOX_SIZE - BORDER_SIZE]
                    overlay_draw.rectangle(fill_rect, fill=(*base_color, 128))

    final_image = Image.alpha_composite(image, overlay_image)

    draw = ImageDraw.Draw(final_image)
    try:
        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        font = ImageFont.load_default()

    text = f"Value: {value_from_third_column:.2f} for {chosen_entry}"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = (final_image.width - text_width - 10, final_image.height - text_height - 10)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    output_filename = f"color_coded_map_{chosen_entry}.png"
    output_path = os.path.join(app.config['STATIC_FOLDER'], output_filename)
    final_image.save(output_path)

    return render_template('result1.html', image_filename=output_filename)
if __name__ == "__main__":
    app.run(debug=True, port=8080)
