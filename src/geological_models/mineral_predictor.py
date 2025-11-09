import torch
import torch.nn as nn
import numpy as np

class MineralPredictor(nn.Module):
    def __init__(self, input_dim, hidden_layers=None):
        super(MineralPredictor, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 256, 128]
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 8))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MineralPredictionModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_dim = 20
        hidden_layers = self.config.get('geological_models.mineral_prediction.hidden_layers')
        
        self.model = MineralPredictor(input_dim, hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                         lr=self.config.get('geological_models.mineral_prediction.learning_rate'))
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.target_minerals = ['copper', 'gold', 'iron', 'silver', 'zinc', 'lead', 'nickel', 'uranium']
    
    def predict_mineral_potential(self, geological_features, spatial_features, terrain_features):
        features = self._combine_features(geological_features, spatial_features, terrain_features)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features_tensor)
            probabilities = torch.sigmoid(predictions)
        
        mineral_potentials = {}
        for i, mineral in enumerate(self.target_minerals):
            mineral_potentials[mineral] = {
                'potential': float(probabilities[0][i].item()),
                'confidence': self._calculate_confidence(float(probabilities[0][i].item())),
                'exploration_priority': self._calculate_priority(float(probabilities[0][i].item()))
            }
        
        return mineral_potentials
    
    def _combine_features(self, geological_features, spatial_features, terrain_features):
        features = []
        
        if geological_features:
            features.extend([
                geological_features.get('rock_diversity', 0),
                geological_features.get('fault_density', 0),
                geological_features.get('mineral_occurrence', 0),
                geological_features.get('formation_complexity', 0)
            ])
        
        if spatial_features:
            features.extend([
                spatial_features.get('clustering_index', 0),
                spatial_features.get('proximity_to_faults', 0),
                spatial_features.get('spatial_autocorrelation', 0)
            ])
        
        if terrain_features:
            features.extend([
                terrain_features.get('elevation_variance', 0),
                terrain_features.get('slope_mean', 0),
                terrain_features.get('drainage_density', 0)
            ])
        
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _calculate_confidence(self, potential):
        if potential < 0.3:
            return 'low'
        elif potential < 0.6:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_priority(self, potential):
        if potential < 0.2:
            return 'low'
        elif potential < 0.5:
            return 'medium'
        elif potential < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def train(self, feature_sets, mineral_labels, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for features, labels in zip(feature_sets, mineral_labels):
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
    
    def generate_mineral_prospectivity_map(self, region_data, grid_resolution=0.1):
        prospectivity_map = {}
        
        if 'bounds' not in region_data:
            return prospectivity_map
        
        bounds = region_data['bounds']
        min_lat, min_lon, max_lat, max_lon = bounds
        
        lat_points = int((max_lat - min_lat) / grid_resolution) + 1
        lon_points = int((max_lon - min_lon) / grid_resolution) + 1
        
        for i in range(lat_points):
            for j in range(lon_points):
                lat = min_lat + i * grid_resolution
                lon = min_lon + j * grid_resolution
                
                synthetic_features = self._generate_synthetic_cell_features(lat, lon)
                potentials = self.predict_mineral_potential(synthetic_features, {}, {})
                
                cell_key = f"{lat:.4f},{lon:.4f}"
                prospectivity_map[cell_key] = {
                    'coordinates': (lat, lon),
                    'potentials': potentials,
                    'best_mineral': max(potentials.items(), key=lambda x: x[1]['potential'])[0],
                    'max_potential': max(pot.values()['potential'] for pot in potentials.values())
                }
        
        return prospectivity_map
    
    def _generate_synthetic_cell_features(self, lat, lon):
        return {
            'rock_diversity': np.random.uniform(0, 1),
            'fault_density': np.random.uniform(0, 0.1),
            'mineral_occurrence': np.random.uniform(0, 1),
            'formation_complexity': np.random.uniform(0, 1)
        }