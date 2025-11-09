import numpy as np
from scipy import ndimage
import cv2

class TerrainModeler:
    def __init__(self):
        self.config = Config()
    
    def create_dem(self, elevation_data, resolution=None):
        if resolution is None:
            resolution = self.config.get('geospatial.resolution')
        
        if elevation_data is None:
            elevation_data = self._generate_synthetic_dem()
        
        dem = {
            'elevation': elevation_data,
            'resolution': resolution,
            'shape': elevation_data.shape
        }
        
        dem['slope'] = self._calculate_slope(elevation_data, resolution)
        dem['aspect'] = self._calculate_aspect(elevation_data)
        dem['curvature'] = self._calculate_curvature(elevation_data)
        dem['roughness'] = self._calculate_roughness(elevation_data)
        
        return dem
    
    def _calculate_slope(self, elevation, resolution):
        dx, dy = np.gradient(elevation, resolution)
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        return slope
    
    def _calculate_aspect(self, elevation):
        dx, dy = np.gradient(elevation)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect[aspect < 0] += 360
        return aspect
    
    def _calculate_curvature(self, elevation):
        dx, dy = np.gradient(elevation)
        d2x, _ = np.gradient(dx)
        _, d2y = np.gradient(dy)
        
        curvature = d2x + d2y
        return curvature
    
    def _calculate_roughness(self, elevation):
        kernel = np.ones((3, 3))
        mean_elevation = ndimage.convolve(elevation, kernel/9, mode='constant')
        roughness = np.abs(elevation - mean_elevation)
        return roughness
    
    def _generate_synthetic_dem(self):
        size = 100
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        elevation = (
            np.sin(X) * np.cos(Y) +
            0.5 * np.sin(2*X) * np.cos(2*Y) +
            0.2 * np.random.randn(size, size)
        )
        
        return elevation
    
    def extract_terrain_features(self, dem):
        features = {}
        
        elevation = dem['elevation']
        slope = dem['slope']
        aspect = dem['aspect']
        
        features['mean_elevation'] = np.mean(elevation)
        features['std_elevation'] = np.std(elevation)
        features['max_elevation'] = np.max(elevation)
        features['min_elevation'] = np.min(elevation)
        
        features['mean_slope'] = np.mean(slope)
        features['std_slope'] = np.std(slope)
        features['max_slope'] = np.max(slope)
        
        features['aspect_diversity'] = self._calculate_aspect_diversity(aspect)
        features['terrain_ruggedness'] = np.mean(dem['roughness'])
        
        features['drainage_patterns'] = self._analyze_drainage(elevation)
        
        return features
    
    def _calculate_aspect_diversity(self, aspect):
        bins = np.linspace(0, 360, 9)
        hist, _ = np.histogram(aspect, bins=bins)
        diversity = np.sum(hist > 0) / len(bins)
        return diversity
    
    def _analyze_drainage(self, elevation):
        structure = np.ones((3, 3))
        labeled, num_features = ndimage.label(elevation < np.percentile(elevation, 30), structure=structure)
        
        drainage = {
            'basin_count': num_features,
            'basin_areas': [np.sum(labeled == i) for i in range(1, num_features + 1)],
            'drainage_density': num_features / elevation.size
        }
        
        return drainage
    
    def simulate_erosion(self, dem, iterations=10):
        elevation = dem['elevation'].copy()
        
        for _ in range(iterations):
            slope = self._calculate_slope(elevation, 1)
            
            erosion_rate = slope * 0.01
            elevation -= erosion_rate
            
            elevation = ndimage.gaussian_filter(elevation, sigma=0.5)
        
        return elevation
    
    def calculate_visibility(self, dem, viewpoint):
        elevation = dem['elevation']
        height, width = elevation.shape
        
        visibility = np.zeros((height, width), dtype=bool)
        
        vx, vy = viewpoint
        if not (0 <= vx < width and 0 <= vy < height):
            return visibility
        
        for y in range(height):
            for x in range(width):
                if x == vx and y == vy:
                    visibility[y, x] = True
                    continue
                
                dx = x - vx
                dy = y - vy
                distance = max(abs(dx), abs(dy))
                
                if distance == 0:
                    continue
                
                steps = max(abs(dx), abs(dy))
                x_step = dx / steps
                y_step = dy / steps
                
                visible = True
                for i in range(1, int(steps)):
                    check_x = int(vx + i * x_step)
                    check_y = int(vy + i * y_step)
                    
                    if 0 <= check_x < width and 0 <= check_y < height:
                        intermediate_height = elevation[check_y, check_x]
                        viewpoint_height = elevation[vy, vx]
                        target_height = elevation[y, x]
                        
                        max_height = viewpoint_height + (target_height - viewpoint_height) * (i / steps)
                        
                        if intermediate_height > max_height:
                            visible = False
                            break
                
                visibility[y, x] = visible
        
        return visibility