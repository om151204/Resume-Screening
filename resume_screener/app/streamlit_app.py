"""
streamlit_app.py
----------------
Simple Streamlit app for resume screening without imghdr dependency
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.predictor import ResumePredictor

# Page configuration
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("Resume Screener")
st.markdown("### Machine Learning Powered Job Category Classifier")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **How it works:**
    1. Upload your resume (PDF or TXT)
    2. The system extracts text
    3. ML model predicts job category
    4. View results with confidence scores

    **Supported formats:**
    - PDF files
    - Text files (.txt)
    """)
    st.markdown("---")
    st.caption("Built with Streamlit & scikit-learn")


# Load model
@st.cache_resource
def load_predictor():
    try:
        return ResumePredictor()
    except FileNotFoundError:
        st.error("⚠️ Model not found! Please run `python train.py` first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# File upload section
st.subheader("📤 Upload Resume")
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt"],
    help="Upload PDF or TXT file containing resume"
)

# Process uploaded file
if uploaded_file is not None:
    # Read file
    file_bytes = uploaded_file.read()
    file_name = uploaded_file.name
    file_size = len(file_bytes) / 1024  # KB

    # Show file info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", file_name)
    with col2:
        st.metric("File Size", f"{file_size:.1f} KB")
    with col3:
        st.metric("File Type", file_name.split('.')[-1].upper())

    st.markdown("---")

    # Load predictor
    predictor = load_predictor()

    if predictor:
        with st.spinner("🔍 Analyzing resume..."):
            try:
                # Get prediction
                result = predictor.predict_from_file(file_bytes, file_name)

                # Extract results
                category = result["predicted_category"]
                confidence = result["confidence"]
                all_scores = result["all_scores"]
                raw_text = result["raw_text"]

                # Display prediction
                st.subheader("🎯 Prediction Result")

                # Create styled result card
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 2rem;
                        border-radius: 10px;
                        text-align: center;
                        border: 2px solid #4CAF50;
                    ">
                        <h3 style="color: #666;">Predicted Job Category</h3>
                        <h1 style="color: #4CAF50; font-size: 2.5rem;">{category}</h1>
                        <p style="font-size: 1.2rem;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Show top predictions
                st.subheader("📊 Top 10 Predictions")

                # Prepare data for visualization
                top_scores = dict(list(all_scores.items())[:10])
                categories_list = list(top_scores.keys())
                probabilities = [score * 100 for score in top_scores.values()]

                # Create horizontal bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=probabilities,
                        y=categories_list,
                        orientation='h',
                        marker_color=['#4CAF50' if cat == category else '#90CAF9' for cat in categories_list],
                        text=[f"{p:.1f}%" for p in probabilities],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
                    )
                ])

                fig.update_layout(
                    title="Category Probabilities",
                    xaxis_title="Probability (%)",
                    yaxis_title="Job Category",
                    height=500,
                    yaxis={'autorange': 'reversed'},
                    showlegend=False,
                    margin=dict(l=0, r=50, t=50, b=0)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display scores table
                with st.expander("📋 View All Category Scores"):
                    # Create dataframe
                    df_scores = pd.DataFrame([
                        {"Rank": i + 1, "Category": cat, "Score": f"{score * 100:.2f}%"}
                        for i, (cat, score) in enumerate(all_scores.items())
                    ])
                    st.dataframe(df_scores, use_container_width=True, hide_index=True)

                # Text preview
                with st.expander("📝 View Extracted Resume Text"):
                    st.text_area(
                        "Extracted text content:",
                        value=raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""),
                        height=300,
                        disabled=True
                    )
                    st.caption(f"Total characters: {len(raw_text):,} | Words: {len(raw_text.split()):,}")

            except Exception as e:
                st.error(f"❌ Error processing resume: {str(e)}")
                st.info("Please make sure the file is a valid resume with readable text.")
    else:
        st.warning("⚠️ Predictor not available. Please train the model first.")

else:
    # Show placeholder when no file uploaded
    st.info("👆 Upload a resume to get started")

    # Show supported categories
    st.subheader("📋 Supported Job Categories")

    categories = [
        "Java Developer", "Python Developer", "Data Science", "Machine Learning",
        "DevOps Engineer", "Web Designing", "HR", "Civil Engineer",
        "Business Analyst", "SAP Developer", "Blockchain", "ETL Developer",
        "Network Security", "PMO", "Database", "Hadoop",
        "DotNet Developer", "Testing", "Mechanical Engineer", "Sales",
        "Operations Manager", "Arts", "Health and Fitness", "Advocate"
    ]

    # Display categories in a grid
    for i in range(0, len(categories), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(categories):
                cols[j].markdown(f"• {categories[i + j]}")

    # Add example section
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    - For best results, use clear and well-formatted resumes
    - PDF files should be text-based (not scanned images)
    - The model works best with technical and professional resumes
    """)