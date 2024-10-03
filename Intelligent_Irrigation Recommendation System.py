import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from scipy.optimize import minimize
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import shap
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import xgboost as xgb
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

OPENWEATHER_API_KEY = 'YOUR_API_KEY'

# Saudi Arabia Crop Types and Growth Stages
SAUDI_CROP_TYPES = ["dates", "wheat", "barley", "alfalfa", "sorghum", "millet", "maize", "tomato", "pepper", "cucumber", "melon", "watermelon", "onion", "garlic", "potato", "carrot", "lettuce", "spinach", "cabbage", "broccoli", "cauliflower", "eggplant", "okra", "peas", "beans", "lentils", "chickpeas", "rice", "sugarcane", "cotton", "sunflower", "sesame", "alfalfa", "clover", "berseem", "sudan grass", "bahia grass", "bermuda grass", "rhodes grass", "star grass", "buffel grass", "kikuyu grass", "coastal bermuda grass", "bermuda grass", "rhodes grass", "star grass", "buffel grass", "kikuyu grass", "coastal bermuda grass"]
GROWTH_STAGES = ["seedling", "vegetative", "flowering", "fruiting", "maturity"]

def get_real_time_weather_data(lat, lon):
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    if data['cod'] != 200:
        raise Exception(f"Error fetching weather data: {data['message']}")
    return {
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'pressure': data['main']['pressure'],
        'wind_speed': data['wind']['speed'],
        'clouds': data['clouds']['all']
    }

def get_geocoding_data(city_name):
    url = f'http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}'
    response = requests.get(url)
    data = response.json()
    if not data:
        raise Exception(f"Error fetching geocoding data for city: {city_name}")
    return {
        'lat': data[0]['lat'],
        'lon': data[0]['lon']
    }

def get_forecast_data(lat, lon):
    url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    if data['cod'] != '200':
        raise Exception(f"Error fetching forecast data: {data['message']}")
    return data['list']

def load_crop_data(file_path):
    df = pd.read_csv(file_path)
    return df

def get_combined_data(lat, lon, df):
    weather_data = get_real_time_weather_data(lat, lon)
    df['temperature'] = weather_data['temperature']
    df['humidity'] = weather_data['humidity']
    df['pressure'] = weather_data['pressure']
    df['wind_speed'] = weather_data['wind_speed']
    df['clouds'] = weather_data['clouds']
    return df

def preprocess_data(df):
    df = pd.get_dummies(df, columns=['label', 'soil_type', 'water_source_type'])
    return df

def get_feature_names(df):
    base_features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds', 'N', 'P', 'K', 'ph', 'rainfall', 'soil_moisture', 'sunlight_exposure', 'co2_concentration', 'organic_matter', 'irrigation_frequency', 'crop_density', 'pest_pressure', 'fertilizer_usage', 'growth_stage', 'urban_area_proximity', 'frost_risk', 'water_usage_efficiency']
    categorical_features = ['label', 'soil_type', 'water_source_type']
    feature_names = base_features + [col for col in df.columns if any(cat in col for cat in categorical_features)]
    return feature_names

def split_data(df, feature_names):
    X = df[feature_names]
    y = df['water_usage_efficiency']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler)
    ])
    return pipeline

def build_model():
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model = VotingRegressor(estimators=[('xgb', xgb_model), ('rf', rf_model), ('gb', gb_model)])
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R-squared: {r2}')

def save_model(model, pipeline):
    joblib.dump(model, 'irrigation_model.pkl')
    joblib.dump(pipeline, 'pipeline.pkl')

def get_real_time_soil_moisture_data():
    return np.random.uniform(10, 30)

def calculate_etc(et0, kc):
    return et0 * kc

def predict_irrigation_needs(crop_type, growth_stage, lat, lon, feature_names):
    weather_data = get_real_time_weather_data(lat, lon)
    soil_moisture = get_real_time_soil_moisture_data()
    
    # Example ET₀ and Kc values
    et0 = 5  # Example reference evapotranspiration in mm/day
    kc = 0.9  # Example crop coefficient
    etc = calculate_etc(et0, kc)
    
    input_data = {
        'temperature': weather_data['temperature'],
        'humidity': weather_data['humidity'],
        'pressure': weather_data['pressure'],
        'wind_speed': weather_data['wind_speed'],
        'clouds': weather_data['clouds'],
        'soil_moisture': soil_moisture,
        'crop_type_dates': 1 if crop_type == 'dates' else 0,
        'crop_type_wheat': 1 if crop_type == 'wheat' else 0,
        'crop_type_barley': 1 if crop_type == 'barley' else 0,
        'growth_stage_seedling': 1 if growth_stage == 'seedling' else 0,
        'growth_stage_vegetative': 1 if growth_stage == 'vegetative' else 0,
        'growth_stage_flowering': 1 if growth_stage == 'flowering' else 0,
        'etc': etc
    }
    input_df = pd.DataFrame([input_data], columns=feature_names + ['etc'])
    pipeline = joblib.load('pipeline.pkl')
    input_scaled = pipeline.transform(input_df)
    model = joblib.load('irrigation_model.pkl')
    irrigation_needed = model.predict(input_scaled)[0]
    st.write(f"Raw Irrigation Needed: {irrigation_needed} mm")
    st.write(f"Justification: The recommended irrigation is based on the following parameters:")
    st.write(f"- Temperature: {weather_data['temperature']}°C")
    st.write(f"- Humidity: {weather_data['humidity']}%")
    st.write(f"- Pressure: {weather_data['pressure']} hPa")
    st.write(f"- Wind Speed: {weather_data['wind_speed']} m/s")
    st.write(f"- Clouds: {weather_data['clouds']}%")
    st.write(f"- Soil Moisture: {soil_moisture}%")
    st.write(f"- Crop Type: {crop_type}")
    st.write(f"- Growth Stage: {growth_stage}")
    st.write(f"- Reference Evapotranspiration (ET₀): {et0} mm/day")
    st.write(f"- Crop Coefficient (Kc): {kc}")
    st.write(f"- Crop Evapotranspiration (ETc): {etc} mm/day")
    return irrigation_needed

def objective_function(irrigation_needed, soil_moisture):
    return irrigation_needed * (1 + (soil_moisture / 100))

def optimize_water_usage(irrigation_needed, soil_moisture):
    space = [Real(0, 100, name='irrigation_needed'), Real(0, 100, name='soil_moisture')]
    @use_named_args(space)
    def objective(**params):
        return objective_function(params['irrigation_needed'], params['soil_moisture'])
    result = gp_minimize(objective, space, n_calls=50, random_state=42)
    st.write(f"Optimization Result: {result.x[0]} mm")
    return result.x[0]

def visualize_data(df):
    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Matrix")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    st.subheader("Pairplot")
    sns.pairplot(df)
    st.pyplot(plt)

    st.subheader("Distribution of Features")
    for col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        st.pyplot(plt)

def main():
    st.title("Irrigation Recommendation System")
    crop_type = st.selectbox("Crop Type", SAUDI_CROP_TYPES)
    growth_stage = st.selectbox("Growth Stage", GROWTH_STAGES)
    city_name = st.text_input("City Name", "Medina")
    geocoding_data = get_geocoding_data(city_name)
    lat, lon = geocoding_data['lat'], geocoding_data['lon']
    m = folium.Map(location=[lat, lon], zoom_start=8)
    folium.Marker([lat, lon], popup="Selected Location").add_to(m)
    folium.TileLayer(
        tiles=f'https://tile.openweathermap.org/map/clouds_new/8/130/95.png?appid={OPENWEATHER_API_KEY}',
        attr='OpenWeatherMap',
        name='Clouds',
        overlay=True,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles=f'https://tile.openweathermap.org/map/precipitation_new/8/130/95.png?appid={OPENWEATHER_API_KEY}',
        attr='OpenWeatherMap',
        name='Precipitation',
        overlay=True,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles=f'https://tile.openweathermap.org/map/pressure_new/8/130/95.png?appid={OPENWEATHER_API_KEY}',
        attr='OpenWeatherMap',
        name='Sea Level Pressure',
        overlay=True,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles=f'https://tile.openweathermap.org/map/wind_new/8/130/95.png?appid={OPENWEATHER_API_KEY}',
        attr='OpenWeatherMap',
        name='Wind Speed',
        overlay=True,
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles=f'https://tile.openweathermap.org/map/temp_new/8/130/95.png?appid={OPENWEATHER_API_KEY}',
        attr='OpenWeatherMap',
        name='Temperature',
        overlay=True,
        control=True
    ).add_to(m)
    draw = Draw(export=True)
    draw.add_to(m)
    drawn_features = st_folium(m, width=700, height=500)
    if drawn_features and 'last_active_drawing' in drawn_features:
        last_active_drawing = drawn_features['last_active_drawing']
        if last_active_drawing is not None:
            polygon_coordinates = last_active_drawing['geometry']['coordinates'][0]
            st.write("Selected Polygon Coordinates:", polygon_coordinates)
            lat, lon = np.mean(polygon_coordinates, axis=0)
            st.write(f"Centroid Latitude: {lat}, Longitude: {lon}")
            polygon = Polygon(polygon_coordinates)
            area_size = polygon.area
            st.write(f"Area Size: {area_size} square units")
            if st.button("Get Irrigation Recommendation"):
                try:
                    df = load_crop_data('crop_recommendationV2.csv')
                    df = get_combined_data(lat, lon, df)
                    df = preprocess_data(df)
                    feature_names = get_feature_names(df)
                    irrigation_needed = predict_irrigation_needs(crop_type, growth_stage, lat, lon, feature_names)
                    optimized_irrigation = optimize_water_usage(irrigation_needed, get_real_time_soil_moisture_data())
                    st.success(f"Optimized Irrigation Needed: {optimized_irrigation} mm")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("No drawing detected. Please draw a polygon on the map.")
    else:
        st.warning("No drawing detected. Please draw a polygon on the map.")

    # Visualize the dataset
    df = load_crop_data('crop_recommendationV2.csv')
    df = get_combined_data(24.4667, 39.6, df)
    df = preprocess_data(df)
    visualize_data(df)

if __name__ == '__main__':
    df = load_crop_data('crop_recommendationV2.csv')
    df = get_combined_data(24.4667, 39.6, df)
    df = preprocess_data(df)
    df['etc'] = np.random.uniform(0, 10, df.shape[0])  # Add 'etc' feature with random values
    feature_names = get_feature_names(df) + ['etc']  # Add 'etc' to feature names
    X_train, X_test, y_train, y_test = split_data(df, feature_names)
    pipeline = build_pipeline()
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    model = build_model()
    model = train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, pipeline)
    main()
