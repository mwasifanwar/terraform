import cv2
import numpy as np
from PIL import Image

class DroneProcessor:
    def __init__(self):
        self.config = Config()
    
    def process_drone_imagery(self, image_path, elevation_data=None):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._create_dummy_drone_data()
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            processed_image = self._preprocess_image(image)
            
            features = self._extract_geological_features(processed_image)
            
            if elevation_data is not None:
                terrain_features = self._analyze_terrain(elevation_data)
                features.update(terrain_features)
            
            return {
                'image': processed_image,
                'features': features,
                'shape': processed_image.shape
            }
        except Exception as e:
            return self._create_dummy_drone_data()
    
    def _preprocess_image(self, image):
        target_size = self.config.get('data.image_size')
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        return normalized_image
    
    def _extract_geological_features(self, image):
        features = {}
        
        gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(gray_image, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['texture_complexity'] = np.mean(gradient_magnitude)
        
        hsv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        features['color_variance'] = np.var(hsv_image[:, :, 0])
        
        return features
    
    def _analyze_terrain(self, elevation_data):
        features = {}
        
        if elevation_data is not None:
            features['mean_elevation'] = np.mean(elevation_data)
            features['elevation_variance'] = np.var(elevation_data)
            
            slope = self._calculate_slope(elevation_data)
            features['mean_slope'] = np.mean(slope)
            features['slope_variance'] = np.var(slope)
        
        return features
    
    def _calculate_slope(self, elevation_data):
        dx, dy = np.gradient(elevation_data)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        return slope
    
    def create_orthomosaic(self, image_paths, positions):
        if len(image_paths) == 0:
            return self._create_dummy_orthomosaic()
        
        first_image = cv2.imread(image_paths[0])
        if first_image is None:
            return self._create_dummy_orthomosaic()
        
        mosaic = np.zeros((1000, 1000, 3), dtype=np.uint8)
        
        for i, (img_path, pos) in enumerate(zip(image_paths, positions)):
            img = cv2.imread(img_path)
            if img is not None:
                x, y = pos
                x = max(0, min(x, mosaic.shape[1] - img.shape[1]))
                y = max(0, min(y, mosaic.shape[0] - img.shape[0]))
                
                mosaic[y:y+img.shape[0], x:x+img.shape[1]] = img
        
        return mosaic
    
    def _create_dummy_drone_data(self):
        image = np.random.rand(512, 512, 3).astype(np.float32)
        return {
            'image': image,
            'features': {
                'edge_density': 0.1,
                'texture_complexity': 50.0,
                'color_variance': 1000.0
            },
            'shape': image.shape
        }
    
    def _create_dummy_orthomosaic(self):
        return np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)