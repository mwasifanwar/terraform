import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

class DisasterPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None):
        super(DisasterPredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 5))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DisasterModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = 20
        self.model = DisasterPredictor(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.disaster_types = [
            'landslide', 'earthquake', 'flood', 'volcanic_eruption', 'subsidence'
        ]
    
    def predict_disaster_risk(self, geological_features, terrain_data, historical_data=None):
        self.model.eval()
        
        features = self._extract_features(geological_features, terrain_data, historical_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features_tensor)
            probabilities = torch.sigmoid(predictions)
        
        risk_assessment = {}
        total_risk = 0
        
        for i, disaster_type in enumerate(self.disaster_types):
            risk_score = float(probabilities[0][i].item())
            risk_assessment[disaster_type] = {
                'risk_score': risk_score,
                'risk_level': self._classify_risk_level(risk_score),
                'alert': risk_score > self.config.get('geological_models.disaster_prediction.risk_threshold')
            }
            total_risk += risk_score
        
        risk_assessment['overall_risk'] = total_risk / len(self.disaster_types)
        risk_assessment['highest_risk'] = max(risk_assessment.items(), key=lambda x: x[1]['risk_score'])
        
        return risk_assessment
    
    def _extract_features(self, geological_features, terrain_data, historical_data):
        features = []
        
        if geological_features:
            features.extend([
                geological_features.get('slope_mean', 0),
                geological_features.get('slope_variance', 0),
                geological_features.get('curvature_mean', 0),
                geological_features.get('rock_density', 0),
                geological_features.get('fault_density', 0)
            ])
        
        if terrain_data:
            features.extend([
                terrain_data.get('elevation_mean', 0),
                terrain_data.get('elevation_std', 0),
                terrain_data.get('roughness', 0),
                terrain_data.get('aspect_variance', 0)
            ])
        
        if historical_data:
            features.extend([
                historical_data.get('historical_events', 0),
                historical_data.get('recurrence_interval', 0),
                historical_data.get('magnitude_mean', 0)
            ])
        
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _classify_risk_level(self, risk_score):
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def train(self, feature_sets, disaster_labels, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, labels in zip(feature_sets, disaster_labels):
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                labels_tensor = torch.FloatTensor(labels).unsqueeze(0).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(features_tensor)
                loss = self.criterion(predictions, labels_tensor)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(feature_sets):.4f}")
    
    def generate_risk_map(self, geological_data, terrain_data, region_bounds):
        risk_map = {}
        
        min_lat, min_lon, max_lat, max_lon = region_bounds
        lat_step = (max_lat - min_lat) / 10
        lon_step = (max_lon - min_lon) / 10
        
        for i in range(10):
            for j in range(10):
                lat = min_lat + i * lat_step
                lon = min_lon + j * lon_step
                
                cell_features = self._simulate_cell_features(lat, lon, geological_data, terrain_data)
                risk_assessment = self.predict_disaster_risk(cell_features, {})
                
                risk_map[f"{lat:.4f},{lon:.4f}"] = {
                    'coordinates': (lat, lon),
                    'risks': risk_assessment,
                    'overall_risk': risk_assessment['overall_risk']
                }
        
        return risk_map
    
    def _simulate_cell_features(self, lat, lon, geological_data, terrain_data):
        return {
            'slope_mean': np.random.uniform(0, 45),
            'slope_variance': np.random.uniform(0, 20),
            'curvature_mean': np.random.uniform(-1, 1),
            'rock_density': np.random.uniform(1, 3),
            'fault_density': np.random.uniform(0, 0.1)
        }