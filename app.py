import os
import pandas as pd
import numpy as np
import faiss
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable Streamlit's file watcher for torch modules
os.environ["STREAMLIT_WATCH_EXCLUDE_MODULES"] = "torch,transformers"

# Streamlit configuration
st.set_page_config(page_title="Recipe Generator", layout="wide")

# Load the dataset
@st.cache_data
def load_recipe_data(dataset_path):
    try:
        # Check if the path exists
        if not os.path.exists(dataset_path):
            st.error(f"Dataset not found at: {dataset_path}")
            return None
            
        recipes_df = pd.read_csv(dataset_path)
        required_columns = ['Name', 'RecipeIngredientParts', 'RecipeInstructions']
        if not all(col in recipes_df.columns for col in required_columns):
            st.error(f"Dataset must contain {required_columns} columns.")
            return None
        return recipes_df[required_columns].to_dict('records')
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
        device = "cpu"  # Use CPU to avoid CUDA issues with Streamlit
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
        return pipeline("text-generation", model="gpt2", device=-1)  # Use CPU
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

# Build FAISS index with progress indicator
# Note the leading underscores in the parameter names to prevent hashing
@st.cache_resource(show_spinner=False)
def build_faiss_index(recipes, _tokenizer, _model, _device):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process embeddings in smaller batches
        batch_size = 20
        all_embeddings = []
        total_batches = (len(recipes) + batch_size - 1) // batch_size
        
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i+batch_size]
            batch_texts = [recipe["Name"] + " " + recipe["RecipeIngredientParts"] for recipe in batch]
            
            batch_progress = min(i + batch_size, len(recipes))
            progress_percentage = batch_progress / len(recipes)
            progress_bar.progress(progress_percentage)
            status_text.text(f"Processing recipes: {batch_progress}/{len(recipes)}")
            
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
        
        # Build FAISS index
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
        
        # Format retrieved recipes
        context = ""
        for i, r in enumerate(retrieved_recipes):
            context += f"Recipe {i+1}: {r['Name']}\n"
            context += f"Ingredients: {r['RecipeIngredientParts']}\n"
            context += f"Instructions: {r['RecipeInstructions']}\n\n"
        
        # Create prompt
        input_text = f"""Based on these similar recipes:
{context}
Create a new recipe for: {prompt}
Recipe:"""
        
        # Generate with safer parameters
        output = generator(
            input_text, 
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return output[0]["generated_text"]
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        st.error(f"Failed to generate recipe: {str(e)}")
        return "Error generating recipe. Please try again."

# Main Streamlit app
def main():
    st.title("RAG-Based Recipe Generator")
    st.write("Enter ingredients or a recipe type to generate a custom recipe")
    
    # Default to your actual dataset path
    default_path = r"C:\Users\Lebron James\Pictures\kish\recipes.csv"
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        dataset_path = st.text_input(
            "Dataset Path:", 
            value=default_path
        )
        top_k = st.slider("Number of reference recipes:", 1, 10, 5)
        
        st.write("### System Info")
        st.write(f"Using PyTorch version: {torch.__version__}")
        st.write(f"Using CPU for processing")
    
    # Initial setup
    if not dataset_path:
        st.warning("Please provide a valid path to the recipes dataset")
        return
    
    # Load data
    data_load_state = st.text("Loading recipe data...")
    recipes = load_recipe_data(dataset_path)
    if recipes is None:
        data_load_state.text("Failed to load recipe data")
        return
    data_load_state.text(f"Loaded {len(recipes)} recipes")
    
    # Load models with visible status
    with st.spinner("Loading models (this might take a minute)..."):
        tokenizer, model, device = load_embedding_model()
        if None in (tokenizer, model, device):
            st.error("Failed to load embedding model")
            return
        
        generator = load_generator_model()
        if generator is None:
            st.error("Failed to load generator model")
            return
    
    # Create a session state for the index
    if 'index' not in st.session_state:
        st.session_state.index = None
    
    # Build index with option to skip
    if st.session_state.index is None:
        build_index = st.button("Build Search Index")
        if build_index:
            st.subheader("Building search index")
            st.session_state.index = build_faiss_index(recipes, tokenizer, model, device)
            if st.session_state.index is None:
                st.error("Failed to build search index")
                return
        else:
            st.info("Please build the search index before generating recipes")
            return
    else:
        st.success("Search index is ready")
    
    # Input prompt
    st.subheader("Generate a Recipe")
    prompt = st.text_area("What would you like to cook today?", height=100)
    
    if st.button("Generate Recipe"):
        if not prompt:
            st.warning("Please enter a recipe prompt")
            return
        
        # Generate the recipe
        with st.spinner("Creating your custom recipe..."):
            try:
                # First retrieve similar recipes
                similar_recipes = retrieve_recipes(prompt, st.session_state.index, recipes, tokenizer, model, device, top_k)
                
                if similar_recipes:
                    with st.expander("Similar Recipes Found"):
                        for i, recipe in enumerate(similar_recipes):
                            st.write(f"*{i+1}. {recipe['Name']}*")
                            st.write(f"Ingredients: {recipe['RecipeIngredientParts']}")
                            with st.expander(f"View instructions for {recipe['Name']}"):
                                st.write(recipe['RecipeInstructions'])
                
                    # Generate and display the recipe
                    generated_text = generate_recipe(prompt, generator, st.session_state.index, recipes, tokenizer, model, device)
                    
                    # Extract just the generated recipe part
                    recipe_text = generated_text.split("Recipe:")[1] if "Recipe:" in generated_text else generated_text
                    
                    st.subheader("Your Generated Recipe")
                    st.write(recipe_text)
                    
                    # Download option
                    st.download_button(
                        label="Download Recipe",
                        data=recipe_text,
                        file_name="generated_recipe.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Couldn't find similar recipes. Try a different prompt.")
            except Exception as e:
                st.error(f"An error occurred during generation: {str(e)}")
    
    # Add a reset button
    if st.button("Reset Index"):
        st.session_state.index = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()
