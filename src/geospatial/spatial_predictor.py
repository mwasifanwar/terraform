import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
import geopandas as gpd

class SpatialPredictor:
    def __init__(self):
        self.config = Config()
        self.mineral_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cluster_model = DBSCAN(eps=0.1, min_samples=5)
    
    def predict_mineral_potential(self, geological_data, spatial_features):
        if geological_data is None or spatial_features is None:
            return {}
        
        features = self._extract_prediction_features(geological_data, spatial_features)
        
        if len(features) == 0:
            return {}
        
        predictions = {}
        
        for mineral in ['copper', 'gold', 'iron', 'silver']:
            potential = self._predict_single_mineral(features, mineral)
            predictions[mineral] = potential
        
        return predictions
    
    def _extract_prediction_features(self, geological_data, spatial_features):
        features = {}
        
        if 'processed_data' in geological_data:
            processed = geological_data['processed_data']
            
            if 'rock_distribution' in processed:
                features.update(processed['rock_distribution'])
            
            if 'mineral_distribution' in processed:
                features.update(processed['mineral_distribution'])
        
        if 'spatial_distribution' in spatial_features:
            spatial = spatial_features['spatial_distribution']
            features['total_area'] = spatial.get('area', 0)
            features['extent_area'] = self._calculate_extent_area(spatial.get('extent', [0,0,0,0]))
        
        if 'clustering_patterns' in spatial_features:
            clustering = spatial_features['clustering_patterns']
            features['clustering_index'] = clustering.get('clustering_index', 0)
            features['mean_distance'] = clustering.get('mean_distance', 0)
        
        return features
    
    def _calculate_extent_area(self, extent):
        if len(extent) == 4:
            minx, miny, maxx, maxy = extent
            return (maxx - minx) * (maxy - miny)
        return 0
    
    def _predict_single_mineral(self, features, mineral):
        feature_vector = self._features_to_vector(features)
        
        if len(feature_vector) == 0:
            return 0.0
        
        synthetic_training = np.random.rand(100, len(feature_vector))
        synthetic_target = np.random.rand(100)
        
        self.mineral_model.fit(synthetic_training, synthetic_target)
        
        prediction = self.mineral_model.predict([feature_vector])[0]
        
        return float(np.clip(prediction, 0, 1))
    
    def _features_to_vector(self, features):
        vector = []
        
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                vector.append(value)
        
        return vector
    
    def cluster_geological_features(self, geological_data):
        if geological_data is None or 'geodataframe' not in geological_data:
            return {}
        
        gdf = geological_data['geodataframe']
        
        if len(gdf) == 0:
            return {}
        
        centroids = gdf.geometry.centroid
        coords = np.array([[p.x, p.y] for p in centroids])
        
        if len(coords) < 5:
            return {}
        
        clusters = self.cluster_model.fit_predict(coords)
        
        cluster_analysis = {}
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue
            
            cluster_points = coords[clusters == cluster_id]
            cluster_gdf = gdf[clusters == cluster_id]
            
            cluster_analysis[cluster_id] = {
                'center': np.mean(cluster_points, axis=0).tolist(),
                'size': len(cluster_points),
                'extent': self._calculate_cluster_extent(cluster_points),
                'rock_types': cluster_gdf['rock_type'].value_counts().to_dict() if 'rock_type' in cluster_gdf.columns else {},
                'minerals': cluster_gdf['mineral'].value_counts().to_dict() if 'mineral' in cluster_gdf.columns else {}
            }
        
        return cluster_analysis
    
    def _calculate_cluster_extent(self, points):
        if len(points) == 0:
            return [0, 0, 0, 0]
        
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        
        return [min_x, min_y, max_x, max_y]
    
    def predict_resource_volume(self, geological_data, mineral_type, depth_range=(0, 100)):
        if geological_data is None:
            return 0.0
        
        gdf = geological_data['geodataframe']
        
        if 'mineral' not in gdf.columns or 'geometry' not in gdf.columns:
            return 0.0
        
        mineral_areas = gdf[gdf['mineral'] == mineral_type].geometry.area.sum()
        
        min_depth, max_depth = depth_range
        average_depth = (min_depth + max_depth) / 2
        
        density_factors = {
            'copper': 8.9,
            'gold': 19.3,
            'iron': 7.8,
            'silver': 10.5
        }
        
        density = density_factors.get(mineral_type, 5.0)
        
        volume = mineral_areas * average_depth * density
        
        return float(volume)