"""
Flask web application for knowledge management system.

Provides endpoints for:
- Semantic search
- Cluster exploration
- Topic visualization
- Similar document discovery
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_model, KnowledgeModel
from content_features.extractors import build_embeddings
from vector_index.index import query_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model (load once at startup)
model: KnowledgeModel = None


def init_model(model_path: str = "model_output"):
    """Initialize the global model."""
    global model
    try:
        model = load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.route('/')
def index():
    """Main dashboard page."""
    if model is None:
        return render_template('error.html', message="Model not loaded"), 500
    
    # Get cluster statistics
    hdb_labels = model.cluster_result.hdbscan_labels
    n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    n_noise = list(hdb_labels).count(-1)
    
    stats = {
        'total_documents': len(model.df),
        'hdbscan_clusters': n_clusters,
        'noise_points': n_noise,
        'kmeans_clusters': model.cluster_result.kmeans_model.n_clusters,
        'vector_index_available': model.vector_index.available
    }
    
    return render_template('index.html', stats=stats)


@app.route('/api/search', methods=['POST'])
def search():
    """
    Semantic search endpoint.
    
    Request body:
        {
            "query": "search text",
            "top_k": 10
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    query_text = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    if not query_text:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Generate query embedding
        query_embedding = build_embeddings([query_text], model.content_features.embed_model_name)[0]
        
        # Get other features (use zeros for topics/graph if not available)
        # In practice, you might want to infer topics for the query
        query_features = np.hstack([
            query_embedding,
            np.zeros(model.content_features.lda_topics.shape[1]),
            np.zeros(model.content_features.nmf_topics.shape[1]),
            np.zeros(model.graph_features.feature_matrix.shape[1])
        ])
        
        # Standardize and reduce (using model's scaler and reducer)
        query_scaled = model.fusion_result.scaler.transform(query_features.reshape(1, -1))
        
        # Search
        results = query_index(model.vector_index, query_scaled[0], top_k=top_k)
        
        # Format results
        formatted_results = []
        for doc_id, score in results:
            doc_row = model.df[model.df['doc_id'] == doc_id].iloc[0]
            formatted_results.append({
                'doc_id': doc_id,
                'sender': doc_row.get('sender', ''),
                'subject': doc_row.get('subject', ''),
                'body_preview': doc_row.get('body', '')[:200] + '...',
                'similarity': float(score)
            })
        
        return jsonify({'results': formatted_results})
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """
    Get cluster information.
    
    Query params:
        method: 'hdbscan' or 'kmeans' (default: 'hdbscan')
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    method = request.args.get('method', 'hdbscan')
    
    if method == 'hdbscan':
        labels = model.cluster_result.hdbscan_labels
    else:
        labels = model.cluster_result.kmeans_labels
    
    # Get cluster statistics
    unique_labels = np.unique(labels)
    clusters = []
    
    for label in unique_labels:
        if label == -1 and method == 'hdbscan':
            cluster_name = 'Noise'
        else:
            cluster_name = f'Cluster {int(label)}'
        
        cluster_docs = model.df[labels == label]
        clusters.append({
            'id': int(label),
            'name': cluster_name,
            'size': len(cluster_docs),
            'sample_docs': cluster_docs[['doc_id', 'sender', 'subject']].head(5).to_dict('records')
        })
    
    return jsonify({'clusters': clusters})


@app.route('/api/cluster/<int:cluster_id>', methods=['GET'])
def get_cluster_details(cluster_id: int):
    """
    Get detailed information about a specific cluster.
    
    Query params:
        method: 'hdbscan' or 'kmeans' (default: 'hdbscan')
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    method = request.args.get('method', 'hdbscan')
    
    if method == 'hdbscan':
        labels = model.cluster_result.hdbscan_labels
    else:
        labels = model.cluster_result.kmeans_labels
    
    cluster_docs = model.df[labels == cluster_id]
    
    if len(cluster_docs) == 0:
        return jsonify({'error': 'Cluster not found'}), 404
    
    # Get top senders
    top_senders = cluster_docs['sender'].value_counts().head(10).to_dict()
    
    # Get sample documents
    sample_docs = cluster_docs[['doc_id', 'sender', 'subject', 'body']].head(20).to_dict('records')
    
    return jsonify({
        'cluster_id': cluster_id,
        'size': len(cluster_docs),
        'top_senders': top_senders,
        'documents': sample_docs
    })


@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get topic information."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    method = request.args.get('method', 'lda')  # 'lda', 'nmf', or 'plsa'
    n_words = int(request.args.get('n_words', 10))
    
    if method == 'lda' and model.content_features.lda_model is not None:
        topic_word = model.content_features.lda_model.components_
        vocab = model.content_features.tfidf_vocab
    elif method == 'nmf' and model.content_features.nmf_model is not None:
        topic_word = model.content_features.nmf_model.components_
        vocab = model.content_features.tfidf_vocab
    elif method == 'plsa' and model.content_features.plsa_topics is not None:
        # For pLSA, we need to get P(w|z) from the model
        # This would need to be stored during training
        return jsonify({'error': 'pLSA topic words not available'}), 400
    else:
        return jsonify({'error': f'Topic model {method} not available'}), 400
    
    topics = []
    for topic_idx in range(topic_word.shape[0]):
        top_indices = np.argsort(topic_word[topic_idx])[::-1][:n_words]
        top_words = [(vocab[idx], float(topic_word[topic_idx, idx])) for idx in top_indices]
        topics.append({
            'topic_id': topic_idx,
            'top_words': top_words
        })
    
    return jsonify({'topics': topics})


@app.route('/api/similar/<doc_id>', methods=['GET'])
def get_similar_documents(doc_id: str):
    """
    Find documents similar to a given document.
    
    Query params:
        top_k: Number of similar documents (default: 10)
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    top_k = int(request.args.get('top_k', 10))
    
    # Find document index
    doc_idx = model.df[model.df['doc_id'] == doc_id].index
    if len(doc_idx) == 0:
        return jsonify({'error': 'Document not found'}), 404
    
    doc_idx = doc_idx[0]
    
    # Get document vector
    doc_vector = model.fusion_result.X_scaled[doc_idx]
    
    # Search for similar documents
    results = query_index(model.vector_index, doc_vector, top_k=top_k + 1)  # +1 to exclude self
    
    # Filter out the document itself
    similar = [r for r in results if r[0] != doc_id][:top_k]
    
    # Format results
    formatted_results = []
    for similar_doc_id, score in similar:
        doc_row = model.df[model.df['doc_id'] == similar_doc_id].iloc[0]
        formatted_results.append({
            'doc_id': similar_doc_id,
            'sender': doc_row.get('sender', ''),
            'subject': doc_row.get('subject', ''),
            'body_preview': doc_row.get('body', '')[:200] + '...',
            'similarity': float(score)
        })
    
    return jsonify({
        'query_doc_id': doc_id,
        'similar_documents': formatted_results
    })


if __name__ == '__main__':
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model_output"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001
    
    init_model(model_path)
    
    print(f"\n{'='*60}")
    print(f"Flask app starting on http://localhost:{port}")
    print(f"{'='*60}\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)

