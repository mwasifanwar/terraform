import torch
import torch.nn as nn
import numpy as np
import cv2

class FormationAnalyzer(nn.Module):
    def __init__(self):
        super(FormationAnalyzer, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        segmentation = self.decoder(encoded)
        features = self.feature_extractor(encoded)
        return segmentation, features

class FormationModel:
    def __init__(self):
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = FormationAnalyzer().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        self.formation_types = [
            'sedimentary', 'igneous', 'metamorphic', 'fault', 'fold',
            'volcanic', 'intrusive', 'stratified', 'unconformity'
        ]
    
    def analyze_formation(self, image):
        self.model.eval()
        
        image_tensor = self._preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            segmentation, features = self.model(image_tensor)
        
        segmentation_map = segmentation.squeeze().cpu().numpy()
        feature_vector = features.squeeze().cpu().numpy()
        
        formation_analysis = self._classify_formation(feature_vector)
        
        return {
            'segmentation_map': segmentation_map,
            'feature_vector': feature_vector,
            'formation_type': formation_analysis['type'],
            'confidence': formation_analysis['confidence'],
            'boundaries': self._extract_boundaries(segmentation_map)
        }
    
    def _classify_formation(self, features):
        if len(features) != len(self.formation_types):
            features = np.pad(features, (0, max(0, len(self.formation_types) - len(features))))
        
        scores = np.dot(features[:len(self.formation_types)], np.random.randn(len(self.formation_types)))
        probabilities = self._softmax(scores)
        
        best_idx = np.argmax(probabilities)
        
        return {
            'type': self.formation_types[best_idx],
            'confidence': float(probabilities[best_idx]),
            'all_scores': {self.formation_types[i]: float(probabilities[i]) for i in range(len(self.formation_types))}
        }
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _extract_boundaries(self, segmentation_map):
        threshold = self.config.get('computer_vision.formation_analysis.segmentation_threshold')
        binary_map = (segmentation_map > threshold).astype(np.uint8)
        
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boundaries = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                boundaries.append(contour)
        
        return boundaries
    
    def train(self, images, segmentation_masks, epochs=50):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for img, mask in zip(images, segmentation_masks):
                img_tensor = torch.FloatTensor(img).unsqueeze(0).to(self.device)
                mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)
                
                self.optimizer.zero_grad()
                segmentation, _ = self.model(img_tensor)
                loss = self.criterion(segmentation, mask_tensor)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"mwasifanwar Epoch {epoch}, Loss: {total_loss/len(images):.4f}")
    
    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32) / 255.0
            
            target_size = self.config.get('data.image_size')
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
            
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=2)
            
            image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        return image