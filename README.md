# Philippines Killings Heatmap

This project generates an interactive heatmap to visualize the geographic distribution of killings in the Philippines, based on the provided dataset.

## Description

The `killings_heatmap.html` file is an interactive map that displays the locations of killings reported in the `dataset.csv` file. The heatmap is designed to provide a clear visual representation of the areas with the highest concentration of incidents.

### Features:

-   **Interactive and Zoomable:** You can zoom in and out to explore different regions of the map.
-   **Heatmap Layer:** The color intensity of the heatmap corresponds to the density of killings in a particular area.
-   **Marker Clusters:** As you zoom in, the heatmap is replaced by marker clusters that show the number of incidents in a specific area. Zooming in further will reveal the individual markers.

## How to Generate the Heatmap

1.  **Prerequisites:** Ensure you have Python installed, along with the necessary libraries: `pandas`, `folium`, `geopy`, and `certifi`.
2.  **Run the Script:** Execute the `main.py` script.
3.  **View the Output:** The script will generate the `killings_heatmap.html` file in the `visualizations` directory and automatically open it in your default web browser.

## Data Source

The heatmap is generated using the data from the `dataset.csv` file, which contains information about the location of each killing.

## Technologies Used

-   **Python:** The core programming language used for data processing and visualization.
-   **Pandas:** For data manipulation and analysis.
-   **Folium:** To create the interactive map and heatmap.
-   **Geopy:** For geocoding the location data (i.e., converting location names into geographic coordinates).

