import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from geopy.geocoders import Nominatim
import time
from collections import defaultdict
import webbrowser
import os
import ssl
import geopy.geocoders
import certifi

# Set the SSL certificate file for the environment
os.environ['SSL_CERT_FILE'] = certifi.where()

# --- Configuration ---
DATASET_PATH = 'dataset.csv'
GDP_DATA_PATH = 'gdp_data.csv'
OUTPUT_DIR = 'visualizations'

# --- Utility Functions ---
def load_data(path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(path)
    # Convert date column to datetime objects for time series analysis
    df['kill_list_clean_Date'] = pd.to_datetime(df['kill_list_clean_Date'], errors='coerce')
    # Drop rows where date conversion failed
    df.dropna(subset=['kill_list_clean_Date'], inplace=True)
    return df

def load_gdp_data(path):
    """Loads the GDP per capita data."""
    gdp_df = pd.read_csv(path)
    # Basic cleaning of region names to improve matching
    gdp_df['Region'] = gdp_df['Region'].str.strip().str.upper()
    return gdp_df

def create_output_directory(dir_name):
    """Creates the output directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def save_plot(fig, filename, dir_name=OUTPUT_DIR):
    """Saves a matplotlib figure to the output directory."""
    path = os.path.join(dir_name, filename)
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")

def save_map(map_obj, filename, dir_name=OUTPUT_DIR):
    """Saves a folium map to the output directory and opens it."""
    path = os.path.join(dir_name, filename)
    map_obj.save(path)
    print(f"Saved {path}")
    webbrowser.open(f'file://{os.path.realpath(path)}')

# --- Graph Generation Functions ---

def plot_killings_over_time(df):
    """Plots the number of killings over time (monthly)."""
    df.set_index('kill_list_clean_Date', inplace=True)
    monthly_killings = df.resample('M').size()

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    monthly_killings.plot(ax=ax, color='crimson', marker='o', linestyle='-')

    ax.set_title('Number of Killings Over Time (Monthly)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Killings', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)

    # Forecasting attempt: Add a simple moving average
    moving_avg = monthly_killings.rolling(window=3).mean()
    ax.plot(moving_avg, color='royalblue', linestyle='--', label='3-Month Moving Average (Trend)')
    ax.legend()

    save_plot(fig, 'killings_over_time.png')
    plt.close(fig)

def plot_killings_by_region(df):
    """Plots the number of killings by region."""
    region_counts = df['kill_list_clean_Region'].value_counts().nlargest(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=region_counts.values, y=region_counts.index, ax=ax, palette='viridis')

    ax.set_title('Top 15 Regions by Number of Killings', fontsize=18, fontweight='bold')
    ax.set_xlabel('Number of Killings', fontsize=12)
    ax.set_ylabel('Region', fontsize=12)
    ax.bar_label(ax.containers[0], fmt='%d', padding=3)

    save_plot(fig, 'killings_by_region.png')
    plt.close(fig)

def plot_killer_breakdown(df):
    """Plots the breakdown of alleged killers."""
    killer_cols = ['kill_list_clean_IsPolice', 'kill_list_clean_IsHitmen', 'kill_list_clean_IsUnknown']
    killer_counts = df[killer_cols].sum()
    killer_counts.index = ['Police', 'Hitmen', 'Unknown']

    fig, ax = plt.subplots(figsize=(10, 7))
    killer_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90,
                       colors=sns.color_palette('muted'), wedgeprops={'edgecolor': 'white'})

    ax.set_title('Breakdown of Alleged Killers', fontsize=18, fontweight='bold')
    ax.set_ylabel('') # Hide the y-label

    save_plot(fig, 'killer_breakdown.png')
    plt.close(fig)

# --- Heatmap Generation Functions ---

def get_coordinates(locations, cache):
    """Geocodes locations to get latitude and longitude, with caching."""
    geolocator = Nominatim(user_agent="math10_presentation_agent")
    coordinates = {}
    for location in locations:
        if location in cache:
            coordinates[location] = cache[location]
            continue
        try:
            # Rate limit requests to avoid being blocked
            time.sleep(1)
            geo_result = geolocator.geocode(location)
            if geo_result:
                coordinates[location] = (geo_result.latitude, geo_result.longitude)
                cache[location] = (geo_result.latitude, geo_result.longitude)
                print(f"Geocoded: {location} -> {coordinates[location]}")
            else:
                print(f"Could not geocode: {location}")
                coordinates[location] = None
                cache[location] = None
        except Exception as e:
            print(f"Error geocoding {location}: {e}")
            coordinates[location] = None
            cache[location] = None
    return coordinates

from folium.plugins import HeatMap, MarkerCluster

def create_heatmap(df):
    """Creates a more visually appealing heatmap of killing locations."""
    print("\nStarting heatmap generation... This may take a while due to geocoding.")

    # Use a more specific location format for better geocoding results
    df['full_location'] = df['kill_list_clean_City'] + ", " + df['kill_list_clean_Region'] + ", Philippines"

    # --- Caching and Geocoding ---
    cache_file = 'location_cache.json'
    coord_cache = {}
    if os.path.exists(cache_file):
        import json
        with open(cache_file, 'r') as f:
            try:
                coord_cache = json.load(f)
            except json.JSONDecodeError:
                print("Cache file is corrupted. Starting with an empty cache.")

    # Find which locations are not yet in the cache
    all_locations = set(df['full_location'].unique())
    locations_to_geocode = [loc for loc in all_locations if loc not in coord_cache]

    if locations_to_geocode:
        print(f"Geocoding {len(locations_to_geocode)} new locations...")
        newly_coded = get_coordinates(locations_to_geocode, coord_cache)
        coord_cache.update(newly_coded)

        # Save updated cache
        import json
        with open(cache_file, 'w') as f:
            json.dump(coord_cache, f)
    else:
        print("All locations found in cache.")

    # --- Map Generation ---
    heat_data = []
    points = []
    for index, row in df.iterrows():
        loc = row['full_location']
        if loc in coord_cache and coord_cache[loc] is not None:
            coords = coord_cache[loc]
            heat_data.append(coords)
            points.append(coords)

    if not heat_data:
        print("No valid coordinates found to generate a heatmap.")
        return

    # Create map centered on the Philippines with a more appealing tile
    map_center = [12.8797, 121.7740]
    heatmap_map = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB positron")

    # Add a customized heatmap layer
    HeatMap(heat_data,
            radius=15,
            blur=20,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
           ).add_to(heatmap_map)

    # Add a marker cluster layer for individual points
    marker_cluster = MarkerCluster().add_to(heatmap_map)
    for point in points:
        folium.Marker(location=point).add_to(marker_cluster)

    save_map(heatmap_map, 'killings_heatmap.html')

def create_gdp_heatmap(gdp_df):
    """Creates a heatmap based on GDP per capita."""
    print("\nCreating GDP per capita heatmap...")

    # --- Geocoding GDP Regions ---
    cache_file = 'gdp_location_cache.json'
    coord_cache = {}
    if os.path.exists(cache_file):
        import json
        with open(cache_file, 'r') as f:
            try:
                coord_cache = json.load(f)
            except json.JSONDecodeError:
                print("GDP cache file is corrupted. Starting with an empty cache.")

    # Geocode regions that are not in the cache
    gdp_locations_to_geocode = [region for region in gdp_df['Region'] if region not in coord_cache]
    if gdp_locations_to_geocode:
        print(f"Geocoding {len(gdp_locations_to_geocode)} new GDP regions...")
        newly_coded = get_coordinates(gdp_locations_to_geocode, coord_cache)
        coord_cache.update(newly_coded)
        import json
        with open(cache_file, 'w') as f:
            json.dump(coord_cache, f)

    # --- Map Generation ---
    gdp_heat_data = []
    for index, row in gdp_df.iterrows():
        region = row['Region']
        if region in coord_cache and coord_cache[region] is not None:
            # Weight the heatmap by GDP per capita
            gdp_heat_data.append(list(coord_cache[region]) + [row['GDP_per_capita']])

    if not gdp_heat_data:
        print("No valid coordinates for GDP data found.")
        return

    map_center = [12.8797, 121.7740]
    gdp_map = folium.Map(location=map_center, zoom_start=6, tiles="CartoDB dark_matter")

    HeatMap(gdp_heat_data,
            name='GDP per Capita',
            radius=25,
            blur=30,
            gradient={0.2: 'green', 0.5: 'yellow', 1: 'red'}
           ).add_to(gdp_map)

    save_map(gdp_map, 'gdp_by_city.html')

def plot_killings_vs_gdp(df, gdp_df):
    """Plots a comparative chart of killings vs. GDP per capita."""
    print("\nGenerating Killings vs. GDP comparison chart...")

    # Prepare killings data
    region_killings = df['kill_list_clean_Region'].str.upper().value_counts().reset_index()
    region_killings.columns = ['Region', 'Killings']

    # Merge datasets
    merged_df = pd.merge(region_killings, gdp_df, on='Region', how='inner')

    if merged_df.empty:
        print("Could not merge killings and GDP data. Check region names for consistency.")
        return

    # --- Create Plot ---
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # Bar plot for killings
    sns.barplot(x='Region', y='Killings', data=merged_df, ax=ax1, alpha=0.6, color='red')
    ax1.set_ylabel('Number of Killings', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    plt.xticks(rotation=90)

    # Line plot for GDP
    ax2 = ax1.twinx()
    sns.lineplot(x='Region', y='GDP_per_capita', data=merged_df, ax=ax2, color='blue', marker='o')
    ax2.set_ylabel('GDP per Capita (PHP)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.suptitle('Comparison of Killings and GDP per Capita by Region', fontsize=20, fontweight='bold')
    save_plot(fig, 'killings_vs_gdp.png')
    plt.close(fig)

def create_choropleth_map(df):
    """Creates a choropleth map showing the number of killings per region."""
    print("\nCreating choropleth map of killings per region.")
    # This requires a GeoJSON file for the regions of the Philippines.
    # We will use a public one.
    geojson_url = "https://raw.githubusercontent.com/macoymejia/geojsonph/master/ph-regions-provinces-cities-municipalities-barangays.json"

    region_counts = df['kill_list_clean_Region'].value_counts().reset_index()
    region_counts.columns = ['region', 'count']

    # Create map centered on the Philippines
    map_center = [12.8797, 121.7740]
    choropleth_map = folium.Map(location=map_center, zoom_start=6)

    # Add choropleth layer
    folium.Choropleth(
        geo_data=geojson_url,
        name='choropleth',
        data=region_counts,
        columns=['region', 'count'],
        key_on='feature.properties.name', # This key might need adjustment based on the GeoJSON file
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Killings per Region'
    ).add_to(choropleth_map)

    folium.LayerControl().add_to(choropleth_map)

    save_map(choropleth_map, 'killings_choropleth_by_region.html')


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting data visualization generation...")
    create_output_directory(OUTPUT_DIR)

    dataset = load_data(DATASET_PATH)
    gdp_dataset = load_gdp_data(GDP_DATA_PATH)

    # Generate and save plots
    plot_killings_over_time(dataset.copy())
    plot_killings_by_region(dataset.copy())
    plot_killer_breakdown(dataset.copy())
    plot_killings_vs_gdp(dataset.copy(), gdp_dataset.copy())

    # Generate and save maps
    # Note: Heatmap generation can be slow on the first run
    create_heatmap(dataset.copy())
    create_gdp_heatmap(gdp_dataset.copy())
    # create_choropleth_map(dataset.copy()) # This is commented out as it requires a specific geojson and might fail.

    print(f"\nAll visualizations have been saved to the '{OUTPUT_DIR}' directory.")
    print("The interactive maps have been opened in your web browser.")
