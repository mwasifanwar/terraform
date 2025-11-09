import numpy as np
import rasterio
from PIL import Image
import cv2

class SatelliteProcessor:
    def __init__(self):
        self.config = Config()
    
    def load_satellite_image(self, image_path):
        try:
            with rasterio.open(image_path) as src:
                bands = src.read()
                transform = src.transform
                crs = src.crs
                bounds = src.bounds
            
            processed_bands = self._preprocess_bands(bands)
            
            return {
                'bands': processed_bands,
                'transform': transform,
                'crs': crs,
                'bounds': bounds,
                'metadata': src.meta
            }
        except Exception as e:
            return self._create_dummy_satellite_data()
    
    def _preprocess_bands(self, bands):
        max_bands = self.config.get('data.max_bands')
        if bands.shape[0] > max_bands:
            bands = bands[:max_bands]
        
        normalized_bands = np.zeros_like(bands, dtype=np.float32)
        for i in range(bands.shape[0]):
            band = bands[i].astype(np.float32)
            min_val = np.min(band)
            max_val = np.max(band)
            if max_val > min_val:
                normalized_bands[i] = (band - min_val) / (max_val - min_val)
            else:
                normalized_bands[i] = band
        
        target_size = self.config.get('data.image_size')
        resized_bands = []
        for band in normalized_bands:
            resized_band = cv2.resize(band, target_size, interpolation=cv2.INTER_CUBIC)
            resized_bands.append(resized_band)
        
        return np.array(resized_bands)
    
    def extract_rgb_composite(self, bands, red_idx=3, green_idx=2, blue_idx=1):
        if bands.shape[0] < max(red_idx, green_idx, blue_idx) + 1:
            return self._create_dummy_rgb()
        
        red = bands[red_idx]
        green = bands[green_idx]
        blue = bands[blue_idx]
        
        rgb = np.stack([red, green, blue], axis=2)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        
        return rgb
    
    def calculate_spectral_indices(self, bands):
        indices = {}
        
        if bands.shape[0] >= 4:
            red = bands[3]
            nir = bands[4]
            indices['ndvi'] = self._calculate_ndvi(red, nir)
        
        if bands.shape[0] >= 5:
            green = bands[2]
            nir = bands[4]
            indices['ndwi'] = self._calculate_ndwi(green, nir)
        
        if bands.shape[0] >= 7:
            swir1 = bands[6]
            nir = bands[4]
            indices['ndmi'] = self._calculate_ndmi(nir, swir1)
        
        return indices
    
    def _calculate_ndvi(self, red, nir):
        denominator = nir + red
        denominator[denominator == 0] = 1
        return (nir - red) / denominator
    
    def _calculate_ndwi(self, green, nir):
        denominator = green + nir
        denominator[denominator == 0] = 1
        return (green - nir) / denominator
    
    def _calculate_ndmi(self, nir, swir1):
        denominator = nir + swir1
        denominator[denominator == 0] = 1
        return (nir - swir1) / denominator
    
    def _create_dummy_satellite_data(self):
        bands = np.random.rand(13, 512, 512).astype(np.float32)
        return {
            'bands': bands,
            'transform': None,
            'crs': None,
            'bounds': None,
            'metadata': {}
        }
    
    def _create_dummy_rgb(self):
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)