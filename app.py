# Import preprocessing classes
from preprocessing import ResumeDatasetLoader, ResumePreprocessor  # Adjust import path as needed

# Add required imports
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import sys
import os
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add styling
st.set_page_config(
    page_title="Resume Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# In your app.py, update the custom CSS style
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1f77b4;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 500;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .skill-tag {
            background-color: #e3f2fd;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            display: inline-block;
            color: #0d47a1;  /* Dark blue color for text */
            font-weight: 500;  /* Make text slightly bold */
        }
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            color: #2c3e50;  /* Dark color for text */
        }
        /* Add styles for skill category labels */
        .skill-category {
            font-weight: 600;
            color: #1f77b4;
            margin-top: 0.5rem;
            margin-bottom: 0.3rem;
        }
        /* Style for individual skill items */
        .skill-item {
            background-color: #e3f2fd;
            color: #0d47a1;
            padding: 0.3rem 0.6rem;
            border-radius: 0.3rem;
            margin: 0.2rem;
            display: inline-block;
            font-size: 0.9rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

class ResumeAnalysisApp:
    def __init__(self):
        self.model_path = "resume_analysis_model/checkpoints/trained_model_v1"
        self.load_model()

    def load_model(self):
        """Load the trained model and its components"""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                st.error(f"Model directory not found: {self.model_path}")
                st.stop()
        
            # Load resume data first
            st.info("Loading resume dataset...")
            loader = ResumeDatasetLoader()
            df = loader.load_dataset()
            preprocessor = ResumePreprocessor()
            processed_df = preprocessor.process_dataframe(df)
        
            # Convert to format needed for indexing
            resume_data = processed_df.to_dict('records')
        
            # Load the model components
            st.info("Loading model components...")
            self.model = SentenceTransformer(str(model_path / "embedding_model"))
        
            # Initialize FAISS index
            self.dimension = 384  # dimension for all-MiniLM-L6-v2
            self.index = faiss.IndexFlatL2(self.dimension)
        
            # Process and index resumes
            st.info("Indexing resumes...")
            self.metadata = []
            vectors = []
        
            for resume in resume_data:
                embedding = self.model.encode(resume['cleaned_text'], normalize_embeddings=True)
                vectors.append(embedding)
                self.metadata.append({
                    'content': resume['cleaned_text'],
                    'category': resume['category'],
                    'sections': resume['sections'],
                    'skills': resume['skills']
                })
        
            vectors = np.array(vectors).astype('float32')
            self.index.add(vectors)
        
            st.sidebar.success(f"✅ Model loaded successfully! ({len(self.metadata)} resumes indexed)")
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
        

    def search_resumes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant resumes"""
        try:
            st.write("Generating query embedding...")
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
            st.write("Searching index...")
            distances, indices = self.index.search(query_embedding, top_k)
        
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    similarity = 1 / (1 + float(distance))
                    results.append({
                        'content': metadata['content'],
                        'score': float(distance),
                        'similarity': similarity,
                        'metadata': metadata
                    })
        
            st.write(f"Found {len(results)} results")
            return results
        
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []



    def visualize_skills_distribution(self, results: List[Dict[str, Any]]) -> go.Figure:
        """Create skills distribution visualization"""
        # Create a DataFrame for plotting
        skill_data = {
            'Skill Category': [],
            'Count': []
        }
    
        all_skills = {}
        for result in results:
            for skill_type, skills in result['metadata']['skills'].items():
                if skill_type not in all_skills:
                    all_skills[skill_type] = set()
                all_skills[skill_type].update(skills)
    
        for category, skills in all_skills.items():
            # Format category name for display
            category_name = category.replace('_', ' ').title()
            skill_data['Skill Category'].append(category_name)
            skill_data['Count'].append(len(skills))
    
        df = pd.DataFrame(skill_data)
    
        # Create bar chart
        fig = px.bar(
            df,
            x='Skill Category',
            y='Count',
            title="Skills Distribution in Search Results",
            color='Count',
            color_continuous_scale='Viridis'
        )
    
        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Skill Category",
            yaxis_title="Number of Unique Skills",
            showlegend=False,
            xaxis={'tickangle': 45}
        )
    
        return fig

    def run(self):
        """Run the Streamlit application"""
        st.markdown('<h1 class="main-header">Resume Analysis System</h1>', unsafe_allow_html=True)
        
        # Sidebar controls
        st.sidebar.markdown('<h2 class="sub-header">Search Parameters</h2>', unsafe_allow_html=True)
        
        search_query = st.sidebar.text_area(
            "Enter your search query",
            placeholder="e.g., Python developer with machine learning experience",
            help="Enter keywords or specific requirements"
        )
        
        top_k = st.sidebar.slider(
            "Number of results",
            min_value=1,
            max_value=20,
            value=5,
            help="Select how many matching resumes to display"
        )
        
        skill_categories = st.sidebar.multiselect(
            "Filter by skill categories",
            ["Programming Languages", "Frameworks", "Databases", "Tools And Platforms", "Machine Learning"],
            default=["Programming Languages", "Machine Learning"]
        )
        
        min_similarity = st.sidebar.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Filter results by minimum similarity score"
        )
        
        if st.sidebar.button("Search", type="primary"):
            if search_query:
                with st.spinner("Searching..."):
                    results = self.search_resumes(search_query, top_k)
                
                    # Debug the results
                    st.write(f"Total results before filtering: {len(results)}")
                
                    # Lower the default similarity threshold
                    min_similarity = st.sidebar.slider(
                        "Minimum similarity score",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,  # Lower default value
                        step=0.1
                    )
                
                    filtered_results = [
                        r for r in results 
                        if r['similarity'] >= min_similarity
                    ]
                
                    st.write(f"Results after filtering: {len(filtered_results)}")
                
                    if not filtered_results:
                        st.warning("No results found. Try lowering the similarity threshold or modifying your search terms.")
                        return
                
                    # Continue with displaying results...
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown('<h2 class="sub-header">Search Results</h2>', unsafe_allow_html=True)
                        
                        for i, result in enumerate(filtered_results, 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="result-card">
                                    <h3>Match {i} (Similarity: {result['similarity']:.2f})</h3>
                                    <p><strong>Category:</strong> {result['metadata']['category']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display skills
                                st.markdown("##### Relevant Skills")
                                for category in skill_categories:
                                    category_key = category.lower().replace(" ", "_")
                                    skills = result['metadata']['skills'].get(category_key, [])
                                    if skills:
                                        st.markdown(f"**{category}:**")
                                        skill_html = " ".join([
                                            f'<span class="skill-tag">{skill}</span>'
                                            for skill in skills
                                        ])
                                        st.markdown(skill_html, unsafe_allow_html=True)
                                
                                # Display relevant sections
                                with st.expander("View Resume Sections"):
                                    for section, content in result['metadata']['sections'].items():
                                        if content.strip():
                                            st.markdown(f"**{section.title()}**")
                                            st.text(content[:500] + "..." if len(content) > 500 else content)
                    
                    with col2:
                        st.markdown('<h2 class="sub-header">Analytics</h2>', unsafe_allow_html=True)
                        
                        # Similarity distribution
                        similarities = [r['similarity'] for r in filtered_results]
                        fig = px.histogram(
                            similarities,
                            title="Similarity Score Distribution",
                            labels={'value': 'Similarity Score', 'count': 'Count'},
                            color_discrete_sequence=['rgb(31, 119, 180)']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Skills distribution
                        skills_fig = self.visualize_skills_distribution(filtered_results)
                        st.plotly_chart(skills_fig, use_container_width=True)
                        
                        # Key metrics
                        st.markdown('<h3 class="sub-header">Key Metrics</h3>', unsafe_allow_html=True)
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.markdown("""
                            <div class="metric-card">
                                <h4>Average Similarity</h4>
                                <p style="font-size: 1.5rem;">{:.2f}</p>
                            </div>
                            """.format(np.mean(similarities)), unsafe_allow_html=True)
                            
                        with metrics_col2:
                            st.markdown("""
                            <div class="metric-card">
                                <h4>Total Matches</h4>
                                <p style="font-size: 1.5rem;">{}</p>
                            </div>
                            """.format(len(filtered_results)), unsafe_allow_html=True)
            else:
                st.warning("Please enter a search query")

if __name__ == "__main__":
    app = ResumeAnalysisApp()
    app.run()