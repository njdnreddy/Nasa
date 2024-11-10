# app.py
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
import base64
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

app = Flask(__name__)

def load_dataset(url):
    """Load dataset from various sources based on URL"""
    try:
        if url.startswith("https://github.com/"):
            # Load from GitHub
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            return pd.read_csv(raw_url)
        elif "drive.google.com" in url:
            # Load from Google Drive
            file_id = url.split('/')[-2]
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            return pd.read_csv(gdown_url)
        elif "huggingface.co" in url:
            # Load from Hugging Face
            dataset_name = url.split('/')[-1]
            dataset = pd.read_csv(f"https://huggingface.co/datasets/{dataset_name}/resolve/main/train.csv")
            return dataset
        else:
            # Try direct download
            return pd.read_csv(url)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")

def create_visualizations(data, clusters):
    # Create visualizations and convert to base64 strings
    plt.switch_backend('Agg')
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. PCA Scatter Plot
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data)
    axs[0, 0].scatter(pca_components[:, 0], pca_components[:, 1], c=clusters, cmap='viridis', s=50)
    axs[0, 0].set_title('PCA Scatter Plot (Clusters)')
    axs[0, 0].set_xlabel('PC1')
    axs[0, 0].set_ylabel('PC2')

    # 2. Line Chart
    axs[0, 1].plot(data)
    axs[0, 1].set_title('Feature Trends')
    axs[0, 1].set_xlabel('Data Points')
    axs[0, 1].set_ylabel('Feature Values')

    # 3. Cluster Distribution
    unique, counts = np.unique(clusters, return_counts=True)
    axs[1, 0].bar(unique, counts, color='skyblue')
    axs[1, 0].set_title('Cluster Distribution')
    axs[1, 0].set_xlabel('Cluster')
    axs[1, 0].set_ylabel('Count')

    # 4. Correlation Heatmap
    sns.heatmap(data.corr(), ax=axs[1, 1], cmap='coolwarm', annot=True, fmt=".2f")
    axs[1, 1].set_title('Feature Correlation Heatmap')

    plt.tight_layout()
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        dataset_url = data.get('url')
        n_clusters = int(data.get('clusters', 3))
        
        # Load dataset from URL
        df = load_dataset(dataset_url)
        
        # Clean and prepare data
        df = df.dropna(axis=1, how='all')
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64']:
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True)
        
        # Label encode categorical columns
        le = LabelEncoder()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = le.fit_transform(df[column].astype(str))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df)
        
        # Generate visualizations
        visualization = create_visualizations(df, clusters)
        
        # Get basic dataset info
        dataset_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'features': list(df.columns),
            'num_clusters': n_clusters
        }
        
        return jsonify({
            'success': True,
            'visualization': visualization,
            'dataset_info': dataset_info,
            'message': 'Analysis completed successfully!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during analysis: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)