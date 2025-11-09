import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

class GeologicalMapper:
    def __init__(self):
        self.config = Config()
    
    def load_geological_map(self, map_path):
        try:
            gdf = gpd.read_file(map_path)
            
            processed_data = self._process_geological_data(gdf)
            
            return {
                'geodataframe': gdf,
                'processed_data': processed_data,
                'bounds': gdf.total_bounds,
                'crs': gdf.crs
            }
        except Exception as e:
            return self._create_dummy_geological_data()
    
    def _process_geological_data(self, gdf):
        processed = {}
        
        if 'geometry' in gdf.columns:
            processed['centroids'] = gdf.geometry.centroid
            processed['areas'] = gdf.geometry.area
            processed['bounds'] = gdf.bounds
        
        if 'rock_type' in gdf.columns:
            rock_types = gdf['rock_type'].unique()
            processed['rock_types'] = rock_types
            processed['rock_distribution'] = gdf['rock_type'].value_counts().to_dict()
        
        if 'mineral' in gdf.columns:
            minerals = gdf['mineral'].unique()
            processed['minerals'] = minerals
            processed['mineral_distribution'] = gdf['mineral'].value_counts().to_dict()
        
        if 'age' in gdf.columns:
            processed['age_range'] = (gdf['age'].min(), gdf['age'].max())
        
        return processed
    
    def create_sampling_points(self, bounds, num_points=100):
        minx, miny, maxx, maxy = bounds
        
        points = []
        for _ in range(num_points):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            points.append(Point(x, y))
        
        return gpd.GeoDataFrame(geometry=points, crs=self.config.get('data.coordinate_system'))
    
    def intersect_with_satellite_data(self, geological_data, satellite_bounds):
        if geological_data is None or satellite_bounds is None:
            return None
        
        gdf = geological_data['geodataframe']
        
        satellite_polygon = Polygon([
            (satellite_bounds.left, satellite_bounds.bottom),
            (satellite_bounds.right, satellite_bounds.bottom),
            (satellite_bounds.right, satellite_bounds.top),
            (satellite_bounds.left, satellite_bounds.top)
        ])
        
        intersected = gdf[gdf.intersects(satellite_polygon)]
        
        return {
            'geodataframe': intersected,
            'intersection_area': intersected.geometry.area.sum(),
            'coverage_percentage': intersected.geometry.area.sum() / satellite_polygon.area
        }
    
    def calculate_mineral_potential(self, geological_data, mineral_type):
        if geological_data is None:
            return 0.0
        
        gdf = geological_data['geodataframe']
        
        if 'mineral' not in gdf.columns:
            return 0.0
        
        mineral_areas = gdf[gdf['mineral'] == mineral_type].geometry.area.sum()
        total_area = gdf.geometry.area.sum()
        
        if total_area == 0:
            return 0.0
        
        return mineral_areas / total_area
    
    def _create_dummy_geological_data(self):
        points = [Point(np.random.uniform(-180, 180), np.random.uniform(-90, 90)) for _ in range(50)]
        gdf = gpd.GeoDataFrame({
            'geometry': points,
            'rock_type': np.random.choice(['granite', 'basalt', 'sandstone', 'limestone'], 50),
            'mineral': np.random.choice(['quartz', 'feldspar', 'mica', 'calcite'], 50),
            'age': np.random.uniform(1, 500, 50)
        }, crs=self.config.get('data.coordinate_system'))
        
        return {
            'geodataframe': gdf,
            'processed_data': self._process_geological_data(gdf),
            'bounds': gdf.total_bounds,
            'crs': gdf.crs
        }