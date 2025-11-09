import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio import features

class GISAnalyzer:
    def __init__(self):
        self.config = Config()
    
    def analyze_spatial_patterns(self, geological_data, mineral_detections):
        analysis = {}
        
        if geological_data is not None and 'geodataframe' in geological_data:
            gdf = geological_data['geodataframe']
            
            analysis['spatial_distribution'] = self._calculate_spatial_distribution(gdf)
            analysis['clustering_patterns'] = self._analyze_clustering(gdf)
            analysis['proximity_analysis'] = self._proximity_analysis(gdf)
        
        if mineral_detections:
            analysis['mineral_distribution'] = self._analyze_mineral_distribution(mineral_detections)
            analysis['association_rules'] = self._find_mineral_associations(mineral_detections)
        
        return analysis
    
    def _calculate_spatial_distribution(self, gdf):
        distribution = {}
        
        if 'geometry' in gdf.columns:
            centroids = gdf.geometry.centroid
            distribution['centroid'] = [centroids.x.mean(), centroids.y.mean()]
            distribution['extent'] = gdf.total_bounds.tolist()
            distribution['area'] = gdf.geometry.area.sum()
        
        if 'rock_type' in gdf.columns:
            rock_distribution = gdf.groupby('rock_type').geometry.area.sum().to_dict()
            distribution['rock_areas'] = rock_distribution
        
        return distribution
    
    def _analyze_clustering(self, gdf):
        clustering = {}
        
        if len(gdf) > 1:
            centroids = gdf.geometry.centroid
            coords = np.array([[p.x, p.y] for p in centroids])
            
            mean_coord = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - mean_coord, axis=1)
            
            clustering['mean_distance'] = np.mean(distances)
            clustering['std_distance'] = np.std(distances)
            clustering['clustering_index'] = np.mean(distances) / (np.max(distances) + 1e-8)
        
        return clustering
    
    def _proximity_analysis(self, gdf):
        proximity = {}
        
        if len(gdf) > 1:
            centroids = gdf.geometry.centroid
            coords = np.array([[p.x, p.y] for p in centroids])
            
            distances = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)
            
            if distances:
                proximity['min_distance'] = np.min(distances)
                proximity['max_distance'] = np.max(distances)
                proximity['mean_distance'] = np.mean(distances)
        
        return proximity
    
    def _analyze_mineral_distribution(self, mineral_detections):
        distribution = {}
        
        minerals = [det['mineral'] for det in mineral_detections]
        unique_minerals = set(minerals)
        
        for mineral in unique_minerals:
            mineral_detects = [det for det in mineral_detections if det['mineral'] == mineral]
            distribution[mineral] = {
                'count': len(mineral_detects),
                'mean_confidence': np.mean([det['confidence'] for det in mineral_detects]),
                'locations': [det['center'] for det in mineral_detects]
            }
        
        return distribution
    
    def _find_mineral_associations(self, mineral_detections):
        associations = {}
        
        if len(mineral_detections) < 2:
            return associations
        
        coords = np.array([det['center'] for det in mineral_detections])
        minerals = [det['mineral'] for det in mineral_detections]
        
        association_distance = 50
        
        for i in range(len(mineral_detections)):
            for j in range(i + 1, len(mineral_detections)):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < association_distance:
                    mineral_pair = tuple(sorted([minerals[i], minerals[j]]))
                    if mineral_pair not in associations:
                        associations[mineral_pair] = 0
                    associations[mineral_pair] += 1
        
        return associations
    
    def create_heatmap(self, points, bounds, resolution=100):
        minx, miny, maxx, maxy = bounds
        
        heatmap = np.zeros((resolution, resolution))
        
        x_range = maxx - minx
        y_range = maxy - miny
        
        for point in points:
            if hasattr(point, 'x'):
                x, y = point.x, point.y
            else:
                x, y = point
            
            if minx <= x <= maxx and miny <= y <= maxy:
                i = int((x - minx) / x_range * (resolution - 1))
                j = int((y - miny) / y_range * (resolution - 1))
                heatmap[j, i] += 1
        
        return heatmap
    
    def calculate_buffers(self, features, distance):
        buffers = {}
        
        for name, geometry in features.items():
            if hasattr(geometry, 'buffer'):
                buffer_zone = geometry.buffer(distance)
                buffers[name] = buffer_zone
        
        return buffers