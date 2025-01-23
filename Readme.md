# AeroSpectra: Satellite Image NO2 Concentration Mapping 🌍

AeroSpectra is a web application designed to visualize and analyze air quality data by overlaying **NO2 concentration** values onto **satellite images**. Users can upload satellite images and corresponding **NO2 concentration data** (in Excel format), and the app will process the data to create a color-coded map, highlighting varying levels of NO2 concentration. With its intuitive interface and powerful backend, AeroSpectra makes it easier to explore environmental data and visualize air quality patterns.

## 🚀 Features

- **Upload Satellite Images**: Support for common image formats like PNG, JPG, and TIFF.
- **NO2 Concentration Data Processing**: Upload Excel files containing longitude, latitude, and NO2 concentration values.
- **Color-Coded Maps**: Processed images are color-coded based on NO2 levels for an easy-to-understand representation.
- **Image Display with Overlays**: See the processed image overlaid with NO2 concentration data.
- **Download Processed Image**: Download the resulting color-coded image for further use or sharing.

## ⚙️ Prerequisites

Before you get started, ensure you have the following installed:

- **Python 3.6** or higher
- **pip** – Python package installer



🖥️ Usage
Open your browser and navigate to http://127.0.0.1:8080.
Upload your satellite image (PNG, JPG, or TIFF format).
Upload the corresponding Excel file with NO2 concentration data.
The Excel file should contain:
Longitude: Longitude values.
Latitude: Latitude values.
NO2: NO2 concentration values.
The app processes the data and generates a color-coded map, showing different NO2 levels on the image.
View the processed image with overlays.
Optionally, download the processed image for further analysis or sharing.


📂 File Formats
Satellite Image: Supported formats include PNG, JPG, or TIFF.
NO2 Data: The Excel file should have the following columns:
Longitude: Longitude values corresponding to the satellite image.
Latitude: Latitude values corresponding to the satellite image.
NO2: The NO2 concentration at the respective coordinates.


