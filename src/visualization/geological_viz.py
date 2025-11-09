import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class GeologicalVisualizer:
    def __init__(self):
        self.config = Config()
    
    def plot_formation_analysis(self, formation_analysis):
        if not formation_analysis:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'segmentation_map' in formation_analysis:
            axes[0,0].imshow(formation_analysis['segmentation_map'], cmap='viridis')
            axes[0,0].set_title('Formation Segmentation')
            axes[0,0].axis('off')
        
        if 'boundaries' in formation_analysis:
            axes[0,1].imshow(np.zeros_like(formation_analysis['segmentation_map']), cmap='gray')
            for boundary in formation_analysis['boundaries']:
                axes[0,1].plot(boundary[:, 0, 0], boundary[:, 0, 1], 'r-', linewidth=2)
            axes[0,1].set_title('Detected Boundaries')
            axes[0,1].axis('off')
        
        if 'formation_type' in formation_analysis:
            formation_data = formation_analysis.get('all_scores', {})
            if formation_data:
                minerals = list(formation_data.keys())
                scores = list(formation_data.values())
                
                axes[1,0].bar(minerals, scores, color='lightgreen')
                axes[1,0].set_xlabel('Formation Type')
                axes[1,0].set_ylabel('Probability')
                axes[1,0].set_title('Formation Type Probabilities')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        axes[1,1].text(0.1, 0.5, f"Primary Formation: {formation_analysis.get('formation_type', 'Unknown')}\n"
                      f"Confidence: {formation_analysis.get('confidence', 0):.2f}", 
                      fontsize=12, va='center')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_disaster_risk_analysis(self, risk_assessment):
        if not risk_assessment:
            return None
        
        disaster_types = []
        risk_scores = []
        risk_levels = []
        
        for disaster_type, assessment in risk_assessment.items():
            if disaster_type in ['overall_risk', 'highest_risk']:
                continue
            
            disaster_types.append(disaster_type)
            risk_scores.append(assessment['risk_score'])
            risk_levels.append(assessment['risk_level'])
        
        colors = []
        for level in risk_levels:
            if level == 'very_high':
                colors.append('red')
            elif level == 'high':
                colors.append('orange')
            elif level == 'medium':
                colors.append('yellow')
            else:
                colors.append('green')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        bars = ax1.bar(disaster_types, risk_scores, color=colors, alpha=0.7)
        ax1.set_xlabel('Disaster Type')
        ax1.set_ylabel('Risk Score')
        ax1.set_title('Disaster Risk Assessment')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, risk_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        risk_categories = ['very_high', 'high', 'medium', 'low']
        category_counts = [risk_levels.count(cat) for cat in risk_categories]
        
        ax2.pie(category_counts, labels=risk_categories, autopct='%1.1f%%', 
               colors=['red', 'orange', 'yellow', 'green'])
        ax2.set_title('Risk Level Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_mineral_potential_map(self, prospectivity_map):
        if not prospectivity_map:
            return None
        
        lats = []
        lons = []
        potentials = []
        minerals = []
        
        for key, data in prospectivity_map.items():
            lat, lon = data['coordinates']
            lats.append(lat)
            lons.append(lon)
            potentials.append(data['max_potential'])
            minerals.append(data['best_mineral'])
        
        unique_minerals = list(set(minerals))
        mineral_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_minerals)))
        color_map = dict(zip(unique_minerals, mineral_colors))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(lons, lats, c=[color_map[mineral] for mineral in minerals], 
                           s=[pot * 100 for pot in potentials], alpha=0.6)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Mineral Prospectivity Map')
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=color_map[mineral], markersize=8, label=mineral)
                          for mineral in unique_minerals]
        ax.legend(handles=legend_elements, title='Minerals')
        
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_resource_estimates(self, resource_estimates):
        if not resource_estimates:
            return None
        
        minerals = list(resource_estimates.keys())
        masses = [est['estimated_mass_kg'] for est in resource_estimates.values()]
        grades = [est['average_grade_percent'] for est in resource_estimates.values()]
        viabilities = [est['economic_viability'] for est in resource_estimates.values()]
        
        viability_colors = {
            'highly_viable': 'green',
            'potentially_viable': 'orange',
            'not_viable': 'red'
        }
        
        colors = [viability_colors.get(viability, 'gray') for viability in viabilities]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        bars = ax1.bar(minerals, masses, color=colors, alpha=0.7)
        ax1.set_xlabel('Mineral')
        ax1.set_ylabel('Estimated Mass (kg)')
        ax1.set_title('Resource Estimates')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_yscale('log')
        
        for bar, mass in zip(bars, masses):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{mass:.1e}', ha='center', va='bottom', rotation=45)
        
        ax2.bar(minerals, grades, color=colors, alpha=0.7)
        ax2.set_xlabel('Mineral')
        ax2.set_ylabel('Average Grade (%)')
        ax2.set_title('Mineral Grades')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig