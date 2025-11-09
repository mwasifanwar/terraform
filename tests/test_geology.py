import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.geological_models.mineral_predictor import MineralPredictionModel
from src.geological_models.resource_estimator import ResourceEstimator

class TestGeologicalModels(unittest.TestCase):
    def test_mineral_prediction(self):
        predictor = MineralPredictionModel()
        geological_features = {'rock_diversity': 0.5, 'fault_density': 0.1}
        spatial_features = {'clustering_index': 0.3}
        terrain_features = {'elevation_variance': 0.2}
        
        predictions = predictor.predict_mineral_potential(geological_features, spatial_features, terrain_features)
        self.assertIn('copper', predictions)
        self.assertIn('gold', predictions)
        self.assertIn('potential', predictions['copper'])
    
    def test_resource_estimation(self):
        estimator = ResourceEstimator()
        mineral_potentials = {'copper': {'potential': 0.7}, 'gold': {'potential': 0.3}}
        area_data = {'area': 1000000, 'depth': 100}
        
        estimates = estimator.estimate_resources(mineral_potentials, {}, area_data)
        self.assertIn('copper', estimates)
        self.assertIn('estimated_mass_kg', estimates['copper'])

if __name__ == '__main__':
    unittest.main()