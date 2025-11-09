import numpy as np
import math

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def coordinates_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def normalize_spectral_bands(bands):
    normalized = np.zeros_like(bands, dtype=np.float32)
    for i in range(bands.shape[0]):
        band = bands[i]
        min_val = np.min(band)
        max_val = np.max(band)
        if max_val > min_val:
            normalized[i] = (band - min_val) / (max_val - min_val)
    return normalized

def calculate_ndvi(red_band, nir_band):
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denominator = nir + red
    denominator[denominator == 0] = 1
    return (nir - red) / denominator

def calculate_ndwi(green_band, nir_band):
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denominator = green + nir
    denominator[denominator == 0] = 1
    return (green - nir) / denominator

def create_geological_mask(coordinates, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for coord in coordinates:
        x, y = coord
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            mask[x, y] = 1
    return mask

def calculate_slope(elevation_data):
    dx, dy = np.gradient(elevation_data)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope

def calculate_aspect(elevation_data):
    dx, dy = np.gradient(elevation_data)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect[aspect < 0] += 360
    return aspect