import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, simpledialog
import pandas as pd
import random
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation

# Constants
WHITE_THRESHOLD = 200  # Threshold to define what is considered "white" or near-white
VARIATION_RANGE = 10   # ±10 µg/m³ variation range
BOX_SIZE = 100         # Size of the boxes
RED_THRESHOLD = 150    # Threshold to define what is considered red or shades of red
N_CLUSTERS = 5         # Number of color clusters to find
BORDER_SIZE = 2        # Size of the border around the boxes

# Example of manual NO2 to RGB mapping (modify as needed)
no2_to_color = {
    100: (255, 0, 0),   # 100 µg/m³ corresponds to Red
    50: (0, 255, 0),    # 50 µg/m³ corresponds to Green
    10: (0, 0, 255),    # 10 µg/m³ corresponds to Blue
    75: (255, 255, 0),  # 75 µg/m³ corresponds to Yellow
    # Add more mappings as needed
}

# Function to check if a pixel is near white
def is_near_white(pixel, threshold=WHITE_THRESHOLD):
    return np.mean(pixel[:3]) > threshold

# Function to check if a pixel is red or shades of red
def is_red(pixel, threshold=RED_THRESHOLD):
    r, g, b = pixel[:3]
    return r > threshold and g < threshold and b < threshold

# Function to map NO2 concentrations to RGB values
def no2_to_rgb(no2_concentration):
    closest_no2 = min(no2_to_color.keys(), key=lambda no2: abs(no2 - no2_concentration))
    return no2_to_color[closest_no2]

# Function to generate a random shade of a base color
def get_random_shade(base_color, variation_range):
    return tuple(
        min(max(int(c + random.uniform(-variation_range, variation_range)), 0), 255)
        for c in base_color
    )

# Function to cluster colors using KMeans
def cluster_colors(image_array, n_clusters=N_CLUSTERS):
    pixels = image_array.reshape(-1, 4)[:, :3]  # Use only RGB values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_.astype(int), kmeans.labels_.reshape(image_array.shape[:2])

# Create a tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Ask the user to select the Excel file containing NO2 data
file_path_excel = filedialog.askopenfilename(title="Select an Excel file", filetypes=[("Excel files", "*.xlsx")])

# Load the Excel file
df = pd.read_excel(file_path_excel)

# Ask the user to input an entry from the first column
chosen_entry = simpledialog.askstring("Input", "Enter an entry from the first column:")

# Filter the DataFrame based on the chosen entry
filtered_df = df[df[df.columns[0]] == chosen_entry]

if filtered_df.empty:
    print(f"No data available for the selected entry: {chosen_entry}")
    exit()

# Ensure there is a third column in the DataFrame
if len(filtered_df.columns) < 3:
    print("Error: The Excel file does not have a third column.")
    exit()

# Extract the value from the third column for the selected entry
value_from_third_column = filtered_df.iloc[0, 2]

# Ask the user to upload the original image file for mapping
file_path_image = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpeg;*.jpg;*.png;*.bmp;*.tiff")])

# Load the selected image
image = Image.open(file_path_image).convert("RGBA")

# Convert the image to a numpy array
image_array = np.array(image)

# Get the clusters of colors from the image
cluster_colors_list, labels = cluster_colors(image_array)

# Create a new RGBA image for the transparent overlay
overlay_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
overlay_draw = ImageDraw.Draw(overlay_image)

# Get the base color for the value from the third column
base_color = no2_to_rgb(value_from_third_column)

# Create masks for detected red areas
red_mask = np.apply_along_axis(is_red, 2, image_array)

# Create a mask for white areas and dilate it to cover more space
white_mask = np.apply_along_axis(lambda x: is_near_white(x), 2, image_array)
dilated_white_mask = binary_dilation(white_mask, structure=np.ones((BOX_SIZE, BOX_SIZE)))

# Generate red blocks only on white spaces around red pixels
for y in range(0, image.height, BOX_SIZE):
    for x in range(0, image.width, BOX_SIZE):
        # Extract region to check
        region = dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]
        if np.any(region & red_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
            red_block_color = get_random_shade((255, 0, 0), VARIATION_RANGE)
            
            # Draw the border
            border_rect = [x, y, x + BOX_SIZE, y + BOX_SIZE]
            overlay_draw.rectangle(border_rect, outline=(0, 0, 0), width=BORDER_SIZE)  # Border color black
            
            # Draw the filled box
            fill_rect = [x + BORDER_SIZE, y + BORDER_SIZE, x + BOX_SIZE - BORDER_SIZE, y + BOX_SIZE - BORDER_SIZE]
            overlay_draw.rectangle(fill_rect, fill=(*red_block_color, 128))

# Fill the remaining white spaces with the color picked from Excel
for y in range(0, image.height, BOX_SIZE):
    for x in range(0, image.width, BOX_SIZE):
        if np.all(dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
            # Ensure it's not already filled with red blocks
            if not np.any(dilated_white_mask[y:y + BOX_SIZE, x:x + BOX_SIZE] & red_mask[y:y + BOX_SIZE, x:x + BOX_SIZE]):
                # Draw the border
                border_rect = [x, y, x + BOX_SIZE, y + BOX_SIZE]
                overlay_draw.rectangle(border_rect, outline=(0, 0, 0), width=BORDER_SIZE)  # Border color black
                
                # Draw the filled box
                fill_rect = [x + BORDER_SIZE, y + BORDER_SIZE, x + BOX_SIZE - BORDER_SIZE, y + BOX_SIZE - BORDER_SIZE]
                overlay_draw.rectangle(fill_rect, fill=(*base_color, 128))

# Overlay the transparent image on the original image
final_image = Image.alpha_composite(image, overlay_image)

# Draw the value from the third column on the final image
draw = ImageDraw.Draw(final_image)

# Load a font that supports Unicode characters
try:
    font_path = "arial.ttf"  # Adjust the path to the font file if needed
    font = ImageFont.truetype(font_path, size=24)
except IOError:
    font = ImageFont.load_default()

# Prepare the text to display
text = f"Value: {value_from_third_column:.2f} for {chosen_entry}"

# Calculate text size using textbbox()
text_bbox = draw.textbbox((0, 0), text, font=font)
text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

# Position the text in the bottom-right corner
text_position = (final_image.width - text_width - 10, final_image.height - text_height - 10)
draw.text(text_position, text, font=font, fill=(255, 255, 255))

# Show the new color-coded image
final_image.show()

# Save the new image
new_image_file_path = f"color_coded_map_{chosen_entry}.png"
final_image.save(new_image_file_path)
print(f"New color-coded map saved as {new_image_file_path}")
