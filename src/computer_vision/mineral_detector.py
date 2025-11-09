import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

class MineralDetector(nn.Module):
    def __init__(self, num_classes=10):
        super(MineralDetector, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.mineral_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.spectral_analyzer = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.fusion_layer = nn.Linear(2048 + 16, num_classes)
    
    def forward(self, image, spectral_data):
        image_features = self.backbone(image)
        
        spectral_features = self.spectral_analyzer(spectral_data)
        
        combined = torch.cat([image_features, spectral_features], dim=1)
        
        output = self.fusion_layer(combined)
        
        return output

class MineralDetectionModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MineralDetector().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.mineral_classes = [
            'quartz', 'feldspar', 'mica', 'calcite', 'dolomite',
            'gypsum', 'halite', 'hematite', 'magnetite', 'other'
        ]
    
    def detect_minerals(self, image, spectral_bands):
        self.model.eval()
        
        image_tensor = self._preprocess_image(image).to(self.device)
        spectral_tensor = self._preprocess_spectral(spectral_bands).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor, spectral_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, class_idx = torch.max(probabilities, dim=1)
        
        mineral_class = self.mineral_classes[class_idx.item()]
        confidence_score = confidence.item()
        
        return {
            'mineral': mineral_class,
            'confidence': confidence_score,
            'all_probabilities': {
                self.mineral_classes[i]: float(probabilities[0][i].item())
                for i in range(len(self.mineral_classes))
            }
        }
    
    def detect_minerals_in_region(self, satellite_data, region_mask=None):
        if satellite_data is None:
            return []
        
        rgb_image = satellite_data.get('rgb', None)
        spectral_bands = satellite_data.get('bands', None)
        
        if rgb_image is None or spectral_bands is None:
            return []
        
        detections = []
        
        if region_mask is None:
            region_mask = np.ones(rgb_image.shape[:2], dtype=bool)
        
        height, width = rgb_image.shape[:2]
        patch_size = self.config.get('computer_vision.mineral_detection.model_input_size')[0]
        stride = patch_size // 2
        
        for y in range(0, height - patch_size, stride):
            for x in range(0, width - patch_size, stride):
                if not region_mask[y:y+patch_size, x:x+patch_size].any():
                    continue
                
                image_patch = rgb_image[y:y+patch_size, x:x+patch_size]
                spectral_patch = spectral_bands[:, y:y+patch_size, x:x+patch_size]
                
                detection = self.detect_minerals(image_patch, spectral_patch)
                
                if detection['confidence'] > self.config.get('computer_vision.mineral_detection.confidence_threshold'):
                    detections.append({
                        'mineral': detection['mineral'],
                        'confidence': detection['confidence'],
                        'bbox': [x, y, x+patch_size, y+patch_size],
                        'center': [x + patch_size//2, y + patch_size//2]
                    })
        
        return detections
    
    def train(self, images, spectral_data, mineral_labels, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for img, spec, label in zip(images, spectral_data, mineral_labels):
                img_tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)
                spec_tensor = torch.FloatTensor(spec).unsqueeze(0).to(self.device)
                label_tensor = torch.LongTensor([label]).to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(img_tensor, spec_tensor)
                loss = self.criterion(predictions, label_tensor)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(images):.4f}")
    
    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            target_size = self.config.get('computer_vision.mineral_detection.model_input_size')
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
            
            image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        return image
    
    def _preprocess_spectral(self, spectral_bands):
        if isinstance(spectral_bands, np.ndarray):
            spectral_bands = spectral_bands.astype(np.float32)
            
            if len(spectral_bands.shape) == 3:
                spectral_bands = np.mean(spectral_bands, axis=(1, 2))
            
            spectral_bands = torch.FloatTensor(spectral_bands).unsqueeze(0)
        
        return spectral_bands