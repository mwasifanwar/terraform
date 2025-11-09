import numpy as np
import cv2
from scipy import ndimage

class FaultDetector:
    def __init__(self):
        self.config = Config()
    
    def detect_faults(self, elevation_data, geological_data=None):
        if elevation_data is None:
            return {}
        
        faults_analysis = {}
        
        gradient_magnitude = self._calculate_gradient_magnitude(elevation_data)
        lineaments = self._detect_lineaments(gradient_magnitude)
        fault_candidates = self._identify_fault_candidates(lineaments, elevation_data)
        
        faults_analysis['gradient_magnitude'] = gradient_magnitude
        faults_analysis['lineaments'] = lineaments
        faults_analysis['fault_candidates'] = fault_candidates
        faults_analysis['fault_density'] = self._calculate_fault_density(fault_candidates, elevation_data.shape)
        
        if geological_data is not None:
            faults_analysis['geological_correlation'] = self._correlate_with_geology(fault_candidates, geological_data)
        
        return faults_analysis
    
    def _calculate_gradient_magnitude(self, elevation):
        sobelx = cv2.Sobel(elevation, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(elevation, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return gradient_magnitude
    
    def _detect_lineaments(self, gradient_magnitude):
        lineaments = {}
        
        edges = cv2.Canny((gradient_magnitude * 255).astype(np.uint8), 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            lineaments['detected_lines'] = lines
            lineaments['line_count'] = len(lines)
            lineaments['line_directions'] = self._calculate_line_directions(lines)
        else:
            lineaments['detected_lines'] = []
            lineaments['line_count'] = 0
            lineaments['line_directions'] = []
        
        return lineaments
    
    def _calculate_line_directions(self, lines):
        directions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            directions.append(angle)
        return directions
    
    def _identify_fault_candidates(self, lineaments, elevation_data):
        candidates = []
        
        if 'detected_lines' not in lineaments or len(lineaments['detected_lines']) == 0:
            return candidates
        
        for line in lineaments['detected_lines']:
            x1, y1, x2, y2 = line[0]
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if length < 20:
                continue
            
            line_elevations = []
            steps = int(length)
            for i in range(steps):
                t = i / steps
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                if 0 <= x < elevation_data.shape[1] and 0 <= y < elevation_data.shape[0]:
                    line_elevations.append(elevation_data[y, x])
            
            if len(line_elevations) > 1:
                elevation_variance = np.var(line_elevations)
                
                if elevation_variance > 100:
                    candidates.append({
                        'line': [x1, y1, x2, y2],
                        'length': length,
                        'elevation_variance': elevation_variance,
                        'orientation': np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    })
        
        return candidates
    
    def _calculate_fault_density(self, fault_candidates, data_shape):
        if len(fault_candidates) == 0:
            return 0.0
        
        total_length = sum(candidate['length'] for candidate in fault_candidates)
        total_area = data_shape[0] * data_shape[1]
        
        return total_length / total_area
    
    def _correlate_with_geology(self, fault_candidates, geological_data):
        correlation = {}
        
        if geological_data is None or 'geodataframe' not in geological_data:
            return correlation
        
        gdf = geological_data['geodataframe']
        
        rock_near_faults = []
        mineral_near_faults = []
        
        for candidate in fault_candidates:
            x1, y1, x2, y2 = candidate['line']
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            for _, feature in gdf.iterrows():
                if hasattr(feature.geometry, 'distance'):
                    distance = feature.geometry.distance(Point(midpoint))
                    if distance < 100:
                        if 'rock_type' in feature:
                            rock_near_faults.append(feature['rock_type'])
                        if 'mineral' in feature:
                            mineral_near_faults.append(feature['mineral'])
        
        correlation['nearby_rocks'] = list(set(rock_near_faults))
        correlation['nearby_minerals'] = list(set(mineral_near_faults))
        correlation['fault_rock_association'] = len(rock_near_faults) / len(fault_candidates) if fault_candidates else 0
        
        return correlation
    
    def predict_fault_activity(self, fault_candidates, seismic_data=None):
        activity_predictions = {}
        
        for i, candidate in enumerate(fault_candidates):
            activity_score = self._calculate_fault_activity_score(candidate, seismic_data)
            
            activity_predictions[f"fault_{i}"] = {
                'line': candidate['line'],
                'activity_score': activity_score,
                'activity_level': self._classify_activity_level(activity_score),
                'risk_assessment': self._assess_fault_risk(activity_score, candidate['length'])
            }
        
        return activity_predictions
    
    def _calculate_fault_activity_score(self, fault, seismic_data):
        base_score = fault['elevation_variance'] / 1000
        
        if seismic_data:
            base_score += seismic_data.get('historical_events', 0) * 0.1
        
        length_factor = min(fault['length'] / 100, 1.0)
        base_score *= length_factor
        
        return min(base_score, 1.0)
    
    def _classify_activity_level(self, score):
        if score < 0.2:
            return 'inactive'
        elif score < 0.5:
            return 'low'
        elif score < 0.8:
            return 'moderate'
        else:
            return 'high'
    
    def _assess_fault_risk(self, activity_score, length):
        risk_score = activity_score * (length / 100)
        
        if risk_score < 0.1:
            return 'very_low'
        elif risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'very_high'