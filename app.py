import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import logging
import gdown

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Streamlit's file watcher for torch modules
os.environ["STREAMLIT_WATCH_EXCLUDE_MODULES"] = "torch,transformers"

# Streamlit configuration
st.set_page_config(page_title="Recipe Generator", layout="wide")

# Google Drive File ID
FILE_ID = "11_ELap6BkQmMRBCrbHygTJ2Avhl_1Gv0"
DATASET_FILE = "recipe.csv"

# Function to download and load the dataset
@st.cache_data
def load_recipe_data():
    try:
        # If file doesn't exist, download it
        if not os.path.exists(DATASET_FILE):
            st.info("Downloading dataset, please wait...")
            gdown.download(f"https://drive.google.com/uc?export=download&id={FILE_ID}", DATASET_FILE, quiet=False)

        # Load CSV file
        recipes_df = pd.read_csv(DATASET_FILE, encoding="latin1")

        # Validate required columns
        required_columns = ["Name", "RecipeIngredientParts", "RecipeInstructions"]
        if not all(col in recipes_df.columns for col in required_columns):
            st.error(f"Dataset must contain {required_columns} columns.")
            return None

        return recipes_df[required_columns].to_dict("records")

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load embedding model once
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()
        device = "cpu"
        model = model.to(device)
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        st.error(f"Failed to load embedding model: {str(e)}")
        return None, None, None

# Load text generation model once
@st.cache_resource(show_spinner=False)
def load_generator_model():
    try:
        return pipeline("text-generation", model="gpt2", device=-1)
    except Exception as e:
        logger.error(f"Error loading generator model: {str(e)}")
        st.error(f"Failed to load generator model: {str(e)}")
        return None

# Encode text into embeddings
def encode_text(text, tokenizer, model, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    except Exception as e:
        logger.error(f"Error encoding text: {str(e)}")
        st.error(f"Failed to encode text: {str(e)}")
        return None

# Build FAISS index
@st.cache_resource(show_spinner=False)
def build_faiss_index(recipes, _tokenizer, _model, _device):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        batch_size = 20
        all_embeddings = []
        total_batches = (len(recipes) + batch_size - 1) // batch_size
        
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i+batch_size]
            batch_texts = [recipe["Name"] + " " + recipe["RecipeIngredientParts"] for recipe in batch]
            
            progress_bar.progress((i + batch_size) / len(recipes))
            status_text.text(f"Processing recipes: {min(i + batch_size, len(recipes))}/{len(recipes)}")
            
            batch_embeddings = []
            for text in batch_texts:
                embedding = encode_text(text, _tokenizer, _model, _device)
                if embedding is not None:
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        if not all_embeddings:
            status_text.text("No embeddings were created. Please check your data.")
            return None
            
        recipe_embeddings = np.vstack(all_embeddings).astype(np.float32)
        dimension = recipe_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(recipe_embeddings)
        
        status_text.text("Index built successfully!")
        progress_bar.progress(1.0)
        return index
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        st.error(f"Failed to build search index: {str(e)}")
        return None

# Retrieve relevant recipes
def retrieve_recipes(query, index, recipes, tokenizer, model, device, top_k=5):
    try:
        query_embedding = encode_text(query, tokenizer, model, device)
        if query_embedding is None:
            return []
        distances, indices = index.search(query_embedding, top_k)
        return [recipes[i] for i in indices[0]]
    except Exception as e:
        logger.error(f"Error retrieving recipes: {str(e)}")
        st.error(f"Failed to retrieve similar recipes: {str(e)}")
        return []

# Generate a recipe
def generate_recipe(prompt, generator, index, recipes, tokenizer, model, device):
    try:
        retrieved_recipes = retrieve_recipes(prompt, index, recipes, tokenizer, model, device)
        if not retrieved_recipes:
            return "Could not find similar recipes to generate from."
        
        context = "\n".join([
            f"Recipe {i+1}: {r['Name']}\nIngredients: {r['RecipeIngredientParts']}\nInstructions: {r['RecipeInstructions']}\n"
            for i, r in enumerate(retrieved_recipes)
        ])
        
        input_text = f"Based on these similar recipes:\n{context}\nCreate a new recipe for: {prompt}\nRecipe:"
        output = generator(input_text, max_length=500, num_return_sequences=1, temperature=0.7, top_p=0.9, do_sample=True)
        return output[0]["generated_text"]
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        st.error(f"Failed to generate recipe: {str(e)}")
        return "Error generating recipe."

# Main Streamlit app
def main():
    st.title("RAG-Based Recipe Generator")
    
    st.sidebar.header("Configuration")
    top_k = st.sidebar.slider("Number of reference recipes:", 1, 10, 5)
    
    recipes = load_recipe_data()
    if recipes is None:
        return
    
    tokenizer, model, device = load_embedding_model()
    generator = load_generator_model()
    
    if None in (tokenizer, model, device, generator):
        return
    
    if 'index' not in st.session_state:
        st.session_state.index = None
    
    if st.session_state.index is None:
        if st.button("Build Search Index"):
            st.session_state.index = build_faiss_index(recipes, tokenizer, model, device)

    st.subheader("Generate a Recipe")
    prompt = st.text_area("What would you like to cook today?")
    
    if st.button("Generate Recipe"):
        if not prompt:
            st.warning("Please enter a recipe prompt")
        else:
            generated_text = generate_recipe(prompt, generator, st.session_state.index, recipes, tokenizer, model, device)
            st.subheader("Your Generated Recipe")
            st.write(generated_text)

if __name__ == "__main__":
    main()
