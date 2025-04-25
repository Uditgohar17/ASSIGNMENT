# SHL Assessment Recommender

This is an AI-powered recommendation engine that suggests relevant SHL assessments based on natural language job descriptions or hiring requirements.

Built using:
- *Gradio* for the frontend interface
- *scikit-learn* for TF-IDF and similarity matching
- *pandas* and *numpy* for data processing

---

## Features

- *Natural Language Input*  
  Paste job descriptions or hiring goals, and receive intelligently matched SHL assessments.

- *TF-IDF Similarity Matching*  
  Matches job needs to assessment descriptions using cosine similarity.

- *Constraint Filtering*  
  Takes into account test type and time duration from your query.

---

## How to Use

1. Clone or download the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
