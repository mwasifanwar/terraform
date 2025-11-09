import folium
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MapGenerator:
    def __init__(self):
        self.config = Config()
    
    def create_interactive_map(self, center_lat, center_lon, zoom_start=10):
        geological_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        return geological_map
    
    def add_mineral_locations(self, geological_map, mineral_detections):
        if not mineral_detections:
            return geological_map
        
        for detection in mineral_detections:
            if 'center' in detection:
                lat, lon = detection['center']
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=f"Mineral: {detection['mineral']}<br>Confidence: {detection['confidence']:.2f}",
                    color='red',
                    fill=True,
                    fillColor='red'
                ).add_to(geological_map)
        
        return geological_map
    
    def add_fault_lines(self, geological_map, fault_analysis):
        if not fault_analysis or 'fault_candidates' not in fault_analysis:
            return geological_map
        
        for candidate in fault_analysis['fault_candidates']:
            if 'line' in candidate:
                x1, y1, x2, y2 = candidate['line']
                
                folium.PolyLine(
                    locations=[[y1, x1], [y2, x2]],
                    color='orange',
                    weight=3,
                    popup=f"Fault Line<br>Length: {candidate['length']:.1f}<br>Activity: {candidate.get('activity_level', 'unknown')}"
                ).add_to(geological_map)
        
        return geological_map
    
    def add_risk_zones(self, geological_map, risk_assessment):
        if not risk_assessment:
            return geological_map
        
        for disaster_type, assessment in risk_assessment.items():
            if disaster_type == 'overall_risk' or disaster_type == 'highest_risk':
                continue
            
            risk_score = assessment['risk_score']
            
            if risk_score > 0.7:
                color = 'red'
            elif risk_score > 0.4:
                color = 'orange'
            else:
                color = 'green'
            
            folium.Circle(
                location=[assessment.get('latitude', 0), assessment.get('longitude', 0)],
                radius=assessment.get('radius', 1000),
                popup=f"{disaster_type}<br>Risk: {risk_score:.2f}<br>Level: {assessment['risk_level']}",
                color=color,
                fill=True,
                fillOpacity=0.3
            ).add_to(geological_map)
        
        return geological_map
    
    def create_geological_cross_section(self, elevation_data, geological_layers, length_km=10):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if elevation_data is not None:
            profile = elevation_data[len(elevation_data)//2, :]
            distance = np.linspace(0, length_km, len(profile))
            
            ax.plot(distance, profile, 'k-', linewidth=2, label='Surface')
        
        if geological_layers:
            for i, layer in enumerate(geological_layers):
                depth = layer.get('depth', 0)
                thickness = layer.get('thickness', 100)
                rock_type = layer.get('rock_type', 'Unknown')
                
                ax.fill_between(distance, profile - depth, profile - depth - thickness, 
                              alpha=0.5, label=rock_type)
        
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title('Geological Cross-Section')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_3d_terrain_model(self, elevation_data):
        if elevation_data is None:
            return None
        
        x = np.arange(elevation_data.shape[1])
        y = np.arange(elevation_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        fig = go.Figure(data=[go.Surface(z=elevation_data, x=X, y=Y)])
        
        fig.update_layout(
            title='3D Terrain Model',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Elevation'
            )
        )
        
        return fig
    
    def plot_mineral_distribution(self, mineral_detections):
        if not mineral_detections:
            return None
        
        minerals = [det['mineral'] for det in mineral_detections]
        confidences = [det['confidence'] for det in mineral_detections]
        
        unique_minerals = list(set(minerals))
        mineral_counts = [minerals.count(mineral) for mineral in unique_minerals]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(unique_minerals, mineral_counts, color='skyblue')
        ax1.set_xlabel('Mineral Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Mineral Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2.hist(confidences, bins=20, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Detection Confidence Distribution')
        
        plt.tight_layout()
        return fig