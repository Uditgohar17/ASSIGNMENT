import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Global catalog (load once)
catalog = None

def load_catalog():
    global catalog
    if catalog is None:
        df = pd.read_csv("attached_assets.csv")
        df = df.drop_duplicates()
        catalog = df.to_dict('records')
    return catalog

def calculate_duration_score(test_duration, max_duration):
    if not max_duration:
        return 1.0
    test_mins = int(test_duration.split()[0])
    return 1.0 if test_mins <= max_duration else 0.0

def extract_test_types(query):
    test_types = ['Cognitive', 'Behavioral', 'Language']
    return [t for t in test_types if t.lower() in query.lower()]

def extract_max_duration(query):
    import re
    duration_patterns = [
        r'(\d+)\s*min',
        r'(\d+)\s*minutes',
        r'within\s*(\d+)',
        r'less than\s*(\d+)',
        r'under\s*(\d+)'
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def get_recommendations(query, max_results=10):
    catalog = load_catalog()
    max_duration = extract_max_duration(query)
    desired_test_types = extract_test_types(query)
    documents = [f"{item['name']} {item['description']} {item['test_type']}" for item in catalog]
    documents.append(query)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    
    scored_tests = []
    for idx, item in enumerate(catalog):
        base_score = similarities[idx]
        duration_score = calculate_duration_score(item['duration'], max_duration)
        type_score = 1.2 if not desired_test_types or item['test_type'] in desired_test_types else 0.8
        final_score = base_score * duration_score * type_score
        if final_score > 0:
            scored_tests.append((final_score, item))
    
    scored_tests.sort(reverse=True, key=lambda x: x[0])
    recommendations = []
    seen = set()
    
    for score, item in scored_tests:
        if item['name'] not in seen and len(recommendations) < max_results:
            seen.add(item['name'])
            recommendations.append({
                "assessment_name": item['name'],
                "url": item['url'],
                "remote_testing_support": item['remote'],
                "adaptive_support": item['irt'],
                "duration": item['duration'],
                "test_type": item['test_type']
            })
    
    return recommendations if recommendations else [catalog[0]]

def recommend(query):
    recommendations = get_recommendations(query)
    return {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "recommendations": recommendations,
        "metadata": {
            "total_results": len(recommendations),
            "max_duration_constraint": extract_max_duration(query),
            "test_types_requested": extract_test_types(query)
        }
    }

# ------------ FastAPI Endpoints ------------
class QueryRequest(BaseModel):
    text: str

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/recommend")
async def recommend_api(query: QueryRequest):
    return recommend(query.text)

# ------------ Gradio Interface ------------
iface = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="Enter job description or requirements:", placeholder="Example: I am hiring for Java developers who can collaborate effectively with business teams..."),
    outputs=gr.JSON(),
    title="SHL Assessment Recommender",
    description="Get recommendations for assessments based on job descriptions or requirements."
)

# ------------ Launch Both ------------
def launch_apps():
    iface.launch(share=True)  # Gradio site for demo

if __name__ == "__main__":
    import threading
    threading.Thread(target=launch_apps).start()
    uvicorn.run(app, host="0.0.0.0", port=7860)
