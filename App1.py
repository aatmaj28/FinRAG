# # import streamlit as st
# # import torch
# # import numpy as np
# # from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
# # from transformers import GPTQConfig
# # import faiss
# # import pickle
# # from sentence_transformers import SentenceTransformer, CrossEncoder
# # import time
# # from nltk.tokenize import word_tokenize
# # import nltk
# # import os
# # from bs4 import BeautifulSoup
# # import re
# # import numpy as np
# # import pandas as pd
# # import pickle
# # import faiss
# # from typing import List, Dict, Tuple, Any
# # import networkx as nx
# # from sklearn.metrics.pairwise import cosine_similarity

# # # Download required NLTK resources
# # nltk.download('punkt', quiet=True)
# # nltk.download('punkt_tab')

# # # Download NLTK data for tokenization
# # try:
# #     nltk.data.find('tokenizers/punkt')
# # except LookupError:
# #     nltk.download('punkt')

# # # Set page title and layout
# # st.set_page_config(page_title="10-K Report Analysis RAG System", layout="wide")

# # @st.cache_resource
# # def load_rag_components(company_name):
# #     """
# #     Load the pre-processed RAG components with caching for better performance
# #     """
# #     with st.spinner(f"Loading RAG components for {company_name}..."):
# #         # Load FAISS index
# #         index = faiss.read_index(f"{company_name.lower()}_10k_enhanced_index.faiss")
        
# #         # Load metadata
# #         with open(f"{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
# #             metadata = pickle.load(f)
        
# #         # Load BM25 model
# #         with open(f"{company_name.lower()}_10k_bm25.pkl", "rb") as f:
# #             bm25_model = pickle.load(f)
        
# #         # Load embedding model
# #         embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
# #         # Load Cross Encoder
# #         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
# #         return index, metadata, embedding_model, bm25_model, cross_encoder

# # # @st.cache_resource
# # # def load_llm():
# # #     """
# # #     Load a different LLM model that might be more compatible
# # #     """
# # #     with st.spinner("Loading language model (this may take a moment)..."):
# # #         model_name = "microsoft/phi-2" # A smaller model that might be easier to load
        
# # #         tokenizer = AutoTokenizer.from_pretrained(model_name)
        
# # #         if tokenizer.pad_token is None:
# # #             tokenizer.pad_token = tokenizer.eos_token
        
# # #         model = AutoModelForCausalLM.from_pretrained(
# # #             model_name,
# # #             device_map="auto",
# # #             trust_remote_code=False
# # #         )
        
# # #         generation_config = GenerationConfig.from_pretrained(model_name)
# # #         generation_config.max_new_tokens = 300
# # #         generation_config.do_sample = True
# # #         generation_config.temperature = 0.6
# # #         generation_config.top_p = 0.9
        
# # #         return tokenizer, model, generation_config

# # @st.cache_resource
# # def load_llm():
# #     """
# #     Load the phi-2 model from local directory using GPU
# #     """
# #     with st.spinner("Loading language model from local files..."):
# #         model_path = r"D:\Projects\NLP_RAG_FAISS\Mistral-7B-GPTQ"
        
# #         tokenizer = AutoTokenizer.from_pretrained(model_path)
        
# #         if tokenizer.pad_token is None:
# #             tokenizer.pad_token = tokenizer.eos_token
        
# #         # Explicitly set device_map to use GPU
# #         model = AutoModelForCausalLM.from_pretrained(
# #             model_path,
# #             device_map="cuda", # Use "cuda" instead of "auto" to force GPU usage
# #             trust_remote_code=False
# #         )
        
# #         generation_config = GenerationConfig.from_pretrained(model_path)
# #         generation_config.max_new_tokens = 300
# #         generation_config.do_sample = True
# #         generation_config.temperature = 0.6
# #         generation_config.top_p = 0.9
        
# #         return tokenizer, model, generation_config

# # def mmr(sorted_results, metadata, embedding_model, top_k, diversity=0.3):
# #     """
# #     Apply Maximum Marginal Relevance to reduce redundancy in results
    
# #     Parameters:
# #     - sorted_results: List of (index, score) tuples sorted by relevance
# #     - metadata: Document metadata
# #     - embedding_model: SentenceTransformer model
# #     - top_k: Number of results to return
# #     - diversity: Balance between relevance and diversity (0-1)
    
# #     Returns:
# #     - Reranked list of (index, score) tuples
# #     """
# #     if not sorted_results:
# #         return []
    
# #     # Get embeddings for all results
# #     indices = [idx for idx, _ in sorted_results]
# #     texts = [metadata[idx]['text'] for idx in indices]
# #     embeddings = embedding_model.encode(texts)
    
# #     # Normalize embeddings
# #     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
# #     embeddings = embeddings / norms
    
# #     # Select first result
# #     selected = [sorted_results[0]]
# #     selected_embeddings = [embeddings[0]]
# #     remaining = sorted_results[1:]
# #     remaining_embeddings = embeddings[1:]
    
# #     # Select remaining results using MMR
# #     while len(selected) < top_k and remaining:
# #         # Calculate similarity to already selected documents
# #         similarities = np.dot(remaining_embeddings, np.array(selected_embeddings).T)
# #         max_similarities = np.max(similarities, axis=1)
        
# #         # Calculate MMR scores
# #         mmr_scores = [(1 - diversity) * score - diversity * sim 
# #                       for (_, score), sim in zip(remaining, max_similarities)]
        
# #         # Select document with highest MMR score
# #         next_idx = np.argmax(mmr_scores)
# #         selected.append(remaining[next_idx])
# #         selected_embeddings.append(remaining_embeddings[next_idx])
        
# #         # Remove selected document
# #         remaining.pop(next_idx)
# #         remaining_embeddings = np.delete(remaining_embeddings, next_idx, axis=0)
    
# #     return selected

# # def rerank_results(query, results, metadata, cross_encoder):
# #     """
# #     Rerank results using a cross-encoder model
    
# #     Parameters:
# #     - query: User query
# #     - results: List of (index, score) tuples
# #     - metadata: Document metadata
# #     - cross_encoder: CrossEncoder model
    
# #     Returns:
# #     - Reranked list of (index, score) tuples
# #     """
# #     if not results:
# #         return []
    
# #     # Prepare pairs for cross-encoder
# #     pairs = [[query, metadata[idx]['text']] for idx, _ in results]
    
# #     # Get cross-encoder scores
# #     cross_scores = cross_encoder.predict(pairs)
    
# #     # Create new results with updated scores
# #     reranked = [(idx, float(score)) for (idx, _), score in zip(results, cross_scores)]
    
# #     # Sort by cross-encoder score
# #     reranked.sort(key=lambda x: x[1], reverse=True)
    
# #     return reranked

# # def hybrid_search(query, faiss_index, metadata, embedding_model, bm25_model,
# #                  cross_encoder=None, k=10, lambda_param=0.5, section_filter=None):
# #     """
# #     Perform hybrid search combining semantic and keyword search with MMR and reranking

# #     Parameters:
# #     - query: User's search query
# #     - faiss_index: FAISS index for semantic search
# #     - metadata: Chunk metadata
# #     - embedding_model: SentenceTransformer model for embeddings
# #     - bm25_model: BM25 model for keyword search
# #     - cross_encoder: CrossEncoder model for reranking
# #     - k: Number of results to return
# #     - lambda_param: Balance between semantic and keyword search (0-1)
# #     - section_filter: Optional filter for specific sections
# #     """
# #     # Apply section filter if provided
# #     if section_filter:
# #         filtered_indices = [i for i, chunk in enumerate(metadata)
# #                            if chunk.get('section_title') == section_filter or
# #                               chunk.get('type') == section_filter]

# #         if not filtered_indices:
# #             print(f"No chunks found with section filter: {section_filter}")
# #             return []
# #     else:
# #         filtered_indices = list(range(len(metadata)))

# #     # Step 1: Semantic search with FAISS
# #     query_embedding = embedding_model.encode([query])[0].reshape(1, -1).astype('float32')

# #     semantic_k = min(k * 2, len(filtered_indices))  # Get more results initially for diversity
# #     distances, indices = faiss_index.search(query_embedding, semantic_k)

# #     # For L2 distance, smaller is better - need to convert to similarity scores
# #     max_dist = np.max(distances[0]) + 1  # Add 1 to ensure positive values

# #     # Filter results by section if needed
# #     if section_filter:
# #         semantic_scores = {}
# #         for i, idx in enumerate(indices[0]):
# #             if idx in filtered_indices:
# #                 semantic_scores[idx] = 1 - (distances[0][i] / max_dist)  # Convert distance to similarity score
# #     else:
# #         semantic_scores = {idx: 1 - (distances[0][i] / max_dist) for i, idx in enumerate(indices[0])}

# #     # Step 2: Keyword search with BM25
# #     tokenized_query = word_tokenize(query.lower())
# #     bm25_scores = bm25_model.get_scores(tokenized_query)

# #     # Normalize BM25 scores
# #     if max(bm25_scores) > 0:
# #         bm25_scores = bm25_scores / max(bm25_scores)

# #     # Create dictionary of BM25 scores
# #     keyword_scores = {}
# #     for i in filtered_indices:
# #         keyword_scores[i] = bm25_scores[i]

# #     # Step 3: Combine scores for hybrid search
# #     hybrid_scores = {}
# #     for idx in set(list(semantic_scores.keys()) + list(keyword_scores.keys())):
# #         if idx in filtered_indices:
# #             sem_score = semantic_scores.get(idx, 0)
# #             key_score = keyword_scores.get(idx, 0)
# #             hybrid_scores[idx] = lambda_param * sem_score + (1 - lambda_param) * key_score

# #     # Get top k*2 results for MMR
# #     sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]

# #     # Step 4: Apply Maximum Marginal Relevance to reduce redundancy
# #     mmr_results = mmr(sorted_results, metadata, embedding_model, k)

# #     # Step 5: Rerank results if cross-encoder is provided
# #     if cross_encoder:
# #         reranked_results = rerank_results(query, mmr_results, metadata, cross_encoder)
# #         results = reranked_results
# #     else:
# #         results = [(idx, score) for idx, score in mmr_results]

# #     # Step 6: Add context to results
# #     final_results = []
# #     for idx, score in results[:k]:
# #         chunk = metadata[idx]

# #         # Add surrounding context
# #         if chunk.get('type') == 'text' and chunk.get('chunk_id', 0) > 0 and chunk.get('chunk_id', 0) < len(metadata) - 1:
# #             prev_chunk = metadata[chunk['chunk_id'] - 1]
# #             next_chunk = metadata[chunk['chunk_id'] + 1]

# #             context = {
# #                 'text': chunk['text'],
# #                 'metadata': chunk,
# #                 'score': score,
# #                 'prev_text': prev_chunk.get('text', '')[:200] + '...' if prev_chunk.get('type') == 'text' else '',
# #                 'next_text': next_chunk.get('text', '')[:200] + '...' if next_chunk.get('type') == 'text' else ''
# #             }
# #         else:
# #             context = {
# #                 'text': chunk['text'],
# #                 'metadata': chunk,
# #                 'score': score,
# #                 'prev_text': '',
# #                 'next_text': ''
# #             }

# #         final_results.append(context)

# #     return final_results

# # def select_top_contexts(retrieved_results, num_contexts=2):
# #     """
# #     Select the top N most relevant contexts based on score
# #     """
# #     # Sort results by score in descending order
# #     sorted_results = sorted(retrieved_results, key=lambda x: x['score'], reverse=True)
    
# #     # Select top contexts, ensuring they have a positive relevance score
# #     top_contexts = [
# #         result for result in sorted_results 
# #         if result['score'] > 0  # Only use contexts with positive relevance score
# #     ][:num_contexts]
    
# #     return top_contexts

# # def format_retrieved_context(retrieved_results):
# #     """
# #     Format top 2 retrieved results into a coherent context string
# #     """
# #     # Select top 2 contexts
# #     selected_contexts = select_top_contexts(retrieved_results)
    
# #     context_parts = []
# #     for i, result in enumerate(selected_contexts, 1):
# #         # Include previous and next text if available
# #         context_text = result['text']
# #         if result.get('prev_text'):
# #             context_text = f"[Previous context: {result['prev_text']}]\n\n{context_text}"
# #         if result.get('next_text'):
# #             context_text = f"{context_text}\n\n[Next context: {result['next_text']}]"
            
# #         context_parts.append(f"Context {i} (Score: {result['score']:.2f}):\n{context_text}")
    
# #     return "\n\n".join(context_parts)

# # # def generate_rag_response(query, retrieved_context, tokenizer, model, generation_config, company_name="NIKE"):
# # #     """
# # #     Generate a response using ONLY the top 2 retrieved contexts
# # #     """
# # #     # Construct the prompt with top 2 contexts in Mistral's instruction format
# # #     prompt = f"""<s>[INST] You are an expert assistant analyzing {company_name}'s 10-K report.
# # # Use ONLY the following two most relevant contexts to answer the query precisely and comprehensively.

# # # Query: {query}

# # # Top 2 Retrieved Contexts:
# # # {retrieved_context}

# # # Important Guidelines:
# # # 1. Base your response STRICTLY on the provided contexts
# # # 2. Do not introduce information not present in these contexts
# # # 3. If the context includes previous or next context indicators, incorporate that information appropriately

# # # Provide a detailed and accurate response: [/INST]"""

# # #     # Prepare inputs with proper attention handling
# # #     inputs = tokenizer(
# # #         prompt, 
# # #         return_tensors="pt", 
# # #         max_length=4096, 
# # #         truncation=True, 
# # #         padding=True
# # #     )
    
# # #     # Move inputs to the same device as the model
# # #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
# # #     # Generate response
# # #     try:
# # #         outputs = model.generate(
# # #             **inputs,
# # #             generation_config=generation_config
# # #         )
        
# # #         # Decode and clean the response
# # #         response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
# # #         return response.strip()
    
# # #     except Exception as e:
# # #         return f"I apologize, but I couldn't generate a complete response based on the given contexts. Error: {e}"

# # def generate_rag_response(query, retrieved_context, tokenizer, model, generation_config, company_name="NIKE"):
# #     """
# #     Generate a response using ONLY the top 2 retrieved contexts
# #     """
# #     # Construct the prompt with top 2 contexts in Mistral's instruction format
# #     prompt = f"""<s>[INST] You are an expert assistant analyzing {company_name}'s 10-K report.
# # Use ONLY the following two most relevant contexts to answer the query precisely and comprehensively.

# # Query: {query}

# # Top 2 Retrieved Contexts:
# # {retrieved_context}

# # Important Guidelines:
# # 1. Base your response STRICTLY on the provided contexts
# # 2. Do not introduce information not present in these contexts
# # 3. If the context includes previous or next context indicators, incorporate that information appropriately
# # 4. Provide ONLY the direct answer without including question format, roleplay, or any other sections
# # 5. Do not use tags like <|Question|>, <|Answer|>, or <|Roleplay|> in your response

# # Provide a detailed and accurate response: [/INST]"""

# #     # Prepare inputs with proper attention handling
# #     inputs = tokenizer(
# #         prompt, 
# #         return_tensors="pt", 
# #         max_length=4096, 
# #         truncation=True, 
# #         padding=True
# #     )
    
# #     # Move inputs to the same device as the model
# #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
# #     # Generate response
# #     try:
# #         outputs = model.generate(
# #             **inputs,
# #             generation_config=generation_config
# #         )
        
# #         # Decode and clean the response
# #         response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
# #         # Extract only the answer part if the response follows the format pattern
# #         if "<|Question|>" in response and "<|Answer|>" in response:
# #             # Extract only the part between <|Answer|> and the next tag (or end of string)
# #             parts = response.split("<|Answer|>")
# #             if len(parts) > 1:
# #                 answer_part = parts[1].split("<|Roleplay|>")[0] if "<|Roleplay|>" in parts[1] else parts[1]
# #                 return answer_part.strip()
        
# #         # Remove any remaining tags just in case
# #         response = response.replace("<|Question|>", "").replace("<|Answer|>", "").replace("<|Roleplay|>", "")
# #         response = response.replace("[INST]", "").replace("[/INST]", "")
        
# #         return response.strip()
    
# #     except Exception as e:
# #         return f"I apologize, but I couldn't generate a complete response based on the given contexts. Error: {e}"
    
# # def rag_pipeline(query, company_name='NIKE', lambda_param=0.5, section_filter=None):
# #     """
# #     Full RAG pipeline: retrieve, generate, and return response
# #     """
# #     # Load RAG components
# #     index, metadata, embedding_model, bm25_model, cross_encoder = load_rag_components(company_name)
    
# #     # Load LLM
# #     tokenizer, model, generation_config = load_llm()
    
# #     # Perform hybrid search
# #     with st.spinner("Retrieving relevant information..."):
# #         retrieved_results = hybrid_search(
# #             query, index, metadata, embedding_model, bm25_model,
# #             cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
# #             section_filter=section_filter
# #         )
    
# #     # Format top 2 contexts
# #     context = format_retrieved_context(retrieved_results)
    
# #     # Generate response
# #     with st.spinner("Generating response..."):
# #         response = generate_rag_response(query, context, tokenizer, model, generation_config, company_name)
    
# #     return {
# #         'query': query,
# #         'retrieved_results': retrieved_results,
# #         'top_2_contexts': context.split('\n\n'),
# #         'response': response
# #     }


# # # Add this right before your main() function
# # def check_gpu():
# #     if torch.cuda.is_available():
# #         gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
# #         memory_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
# #         st.sidebar.success(f"✅ Using GPU\n{gpu_info}\n{memory_info}")
# #     else:
# #         st.sidebar.warning("⚠️ GPU not detected. Using CPU (this will be slow)")

# # def main():
# #     st.title("10-K Report Analysis RAG System")
    
# #     # Custom CSS for better styling
# #     st.markdown("""
# #     <style>
# #     .stApp {
# #         max-width: 1200px;
# #         margin: 0 auto;
# #     }
# #     .result-container {
# #         background-color: #f8f9fa;
# #         padding: 20px;
# #         border-radius: 5px;
# #         margin-bottom: 20px;
# #         color: #333333;  /* Adding explicit dark text color */
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)
    
# #     st.markdown("""
# #     This application allows you to query a company's 10-K report using a Retrieval-Augmented Generation (RAG) system.
# #     The system retrieves relevant information from the report and generates a detailed response to your query.
# #     """)
    
# #     # Sidebar with company selection and advanced settings
# #     st.sidebar.title("Settings")
# #     company_name = st.sidebar.selectbox(
# #         "Select company:",
# #         ["NIKE", "APPLE", "MICROSOFT", "TESLA"],  # Add all available companies
# #         index=0
# #     )
    
# #     # Advanced settings expander in sidebar
# #     with st.sidebar.expander("Advanced Settings"):
# #         lambda_param = st.slider(
# #             "Hybrid search balance (λ):",
# #             min_value=0.0,
# #             max_value=1.0,
# #             value=0.5,
# #             step=0.1,
# #             help="Balance between semantic (1.0) and keyword (0.0) search"
# #         )
        
# #         section_filter = st.text_input(
# #             "Section filter (optional):",
# #             value="",
# #             help="Filter results by section title or type (leave empty for all sections)"
# #         )
        
# #         if not section_filter:
# #             section_filter = None
    
# #     st.sidebar.markdown("---")
# #     st.sidebar.markdown("### Model Information")
# #     st.sidebar.markdown("""
# #     - **LLM**: Mistral 7B Instruct (GPTQ 4-bit)
# #     - **Embedding Model**: all-mpnet-base-v2
# #     - **Re-ranker**: ms-marco-MiniLM-L-6-v2
# #     """)
    
# #     # Main content
# #     st.header(f"Query {company_name}'s 10-K Report")
    
# #     # Example queries
# #     st.markdown("### Example Queries")
# #     example_queries = [
# #         "What are the main revenue sources?",
# #         "Describe the strategy for international markets",
# #         "What are the key financial risks?",
# #         "How does the company approach sustainability?",
# #         "What are the major competitors mentioned?"
# #     ]
    
# #     # Create columns for example query buttons
# #     cols = st.columns(3)
# #     example_buttons = {}
    
# #     for i, query in enumerate(example_queries):
# #         col_idx = i % 3
# #         with cols[col_idx]:
# #             example_buttons[query] = st.button(query)
    
# #     # Query input
# #     query = st.text_area("Enter your query:", height=100, placeholder="Type your question about the company's 10-K report here...")
    
# #     check_gpu()
    
# #     # Set query if example button is clicked
# #     for example_query, clicked in example_buttons.items():
# #         if clicked:
# #             query = example_query
# #             # Need to rerun to update the text area
# #             st.experimental_rerun()
    
# #     # Process button
# #     submit_col1, submit_col2 = st.columns([1, 5])
# #     with submit_col1:
# #         submit_button = st.button("Submit Query", type="primary")
    
# #     if submit_button and query:
# #         start_time = time.time()
        
# #         # Run RAG pipeline
# #         result = rag_pipeline(
# #             query, 
# #             company_name=company_name,
# #             lambda_param=lambda_param,
# #             section_filter=section_filter
# #         )
        
# #         # Display processing time
# #         processing_time = time.time() - start_time
# #         st.info(f"Query processed in {processing_time:.2f} seconds")
        
# #         # # Display results in a nice format
# #         # st.markdown("## Response")
# #         # # st.markdown(f'<div class="result-container">{result["response"]}</div>', unsafe_allow_html=True)
# #         # st.markdown('<div class="result-container">', unsafe_allow_html=True)
# #         # st.write(result["response"])
# #         # st.markdown('</div>', unsafe_allow_html=True)
# #         st.markdown("## Response")
# #         st.info(result["response"])  # Uses Streamlit's native info box with proper styling
        
# #         # Show retrieved contexts in an expander
# #         with st.expander("View retrieved contexts"):
# #             for i, context in enumerate(result['top_2_contexts']):
# #                 st.markdown(f"### {context.split(':', 1)[0]}")
# #                 # st.markdown(context.split(':\n')[1])
# #                 # With this safer version
# #                 parts = context.split(':\n', 1)  # Split on first occurrence only
# #                 if len(parts) > 1:
# #                     st.markdown(parts[1])
# #                 else:
# #                     st.markdown(parts[0])  # Just display the whole context if no split found
# #                 st.markdown("---")
        
# #         # Advanced results analysis
# #         with st.expander("View all retrieved documents with context"):
# #             for i, result_item in enumerate(result['retrieved_results'], 1):
# #                 col1, col2 = st.columns([1, 4])
                
# #                 with col1:
# #                     st.markdown(f"**Score: {result_item['score']:.3f}**")
# #                     if 'metadata' in result_item:
# #                         metadata = result_item['metadata']
# #                         if 'section_title' in metadata:
# #                             st.markdown(f"**Section: {metadata['section_title']}**")
# #                         if 'page' in metadata:
# #                             st.markdown(f"**Page: {metadata['page']}**")
                
# #                 with col2:
# #                     # Display previous context if available
# #                     if result_item.get('prev_text'):
# #                         st.markdown("**Previous Context:**")
# #                         st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['prev_text']}</div>", 
# #                                     unsafe_allow_html=True)
                    
# #                     # Display main text
# #                     st.markdown("**Main Text:**")
# #                     st.markdown(result_item['text'])
                    
# #                     # Display next context if available
# #                     if result_item.get('next_text'):
# #                         st.markdown("**Next Context:**")
# #                         st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['next_text']}</div>", 
# #                                     unsafe_allow_html=True)
                
# #                 st.markdown("---")

# # if __name__ == "__main__":
# #     main()




# import streamlit as st
# import torch
# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import time
# from nltk.tokenize import word_tokenize
# import nltk
# import os
# import requests
# import json
# from bs4 import BeautifulSoup
# import re
# import pandas as pd
# from typing import List, Dict, Tuple, Any
# import networkx as nx
# from sklearn.metrics.pairwise import cosine_similarity

# # Download required NLTK resources
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab')

# # Download NLTK data for tokenization
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Set page title and layout
# st.set_page_config(page_title="10-K Report Analysis RAG System", layout="wide")

# @st.cache_resource
# def load_rag_components(company_name):
#     """
#     Load the pre-processed RAG components with caching for better performance
#     """
#     with st.spinner(f"Loading RAG components for {company_name}..."):
#         # Load FAISS index
#         index = faiss.read_index(f"{company_name.lower()}_10k_enhanced_index.faiss")
        
#         # Load metadata
#         with open(f"{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
#             metadata = pickle.load(f)
        
#         # Load BM25 model
#         with open(f"{company_name.lower()}_10k_bm25.pkl", "rb") as f:
#             bm25_model = pickle.load(f)
        
#         # Load embedding model
#         embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
#         # Load Cross Encoder
#         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
#         return index, metadata, embedding_model, bm25_model, cross_encoder

# def mmr(sorted_results, metadata, embedding_model, top_k, diversity=0.3):
#     """
#     Apply Maximum Marginal Relevance to reduce redundancy in results
    
#     Parameters:
#     - sorted_results: List of (index, score) tuples sorted by relevance
#     - metadata: Document metadata
#     - embedding_model: SentenceTransformer model
#     - top_k: Number of results to return
#     - diversity: Balance between relevance and diversity (0-1)
    
#     Returns:
#     - Reranked list of (index, score) tuples
#     """
#     if not sorted_results:
#         return []
    
#     # Get embeddings for all results
#     indices = [idx for idx, _ in sorted_results]
#     texts = [metadata[idx]['text'] for idx in indices]
#     embeddings = embedding_model.encode(texts)
    
#     # Normalize embeddings
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     embeddings = embeddings / norms
    
#     # Select first result
#     selected = [sorted_results[0]]
#     selected_embeddings = [embeddings[0]]
#     remaining = sorted_results[1:]
#     remaining_embeddings = embeddings[1:]
    
#     # Select remaining results using MMR
#     while len(selected) < top_k and remaining:
#         # Calculate similarity to already selected documents
#         similarities = np.dot(remaining_embeddings, np.array(selected_embeddings).T)
#         max_similarities = np.max(similarities, axis=1)
        
#         # Calculate MMR scores
#         mmr_scores = [(1 - diversity) * score - diversity * sim 
#                       for (_, score), sim in zip(remaining, max_similarities)]
        
#         # Select document with highest MMR score
#         next_idx = np.argmax(mmr_scores)
#         selected.append(remaining[next_idx])
#         selected_embeddings.append(remaining_embeddings[next_idx])
        
#         # Remove selected document
#         remaining.pop(next_idx)
#         remaining_embeddings = np.delete(remaining_embeddings, next_idx, axis=0)
    
#     return selected

# def rerank_results(query, results, metadata, cross_encoder):
#     """
#     Rerank results using a cross-encoder model
    
#     Parameters:
#     - query: User query
#     - results: List of (index, score) tuples
#     - metadata: Document metadata
#     - cross_encoder: CrossEncoder model
    
#     Returns:
#     - Reranked list of (index, score) tuples
#     """
#     if not results:
#         return []
    
#     # Prepare pairs for cross-encoder
#     pairs = [[query, metadata[idx]['text']] for idx, _ in results]
    
#     # Get cross-encoder scores
#     cross_scores = cross_encoder.predict(pairs)
    
#     # Create new results with updated scores
#     reranked = [(idx, float(score)) for (idx, _), score in zip(results, cross_scores)]
    
#     # Sort by cross-encoder score
#     reranked.sort(key=lambda x: x[1], reverse=True)
    
#     return reranked

# def hybrid_search(query, faiss_index, metadata, embedding_model, bm25_model,
#                  cross_encoder=None, k=10, lambda_param=0.5, section_filter=None):
#     """
#     Perform hybrid search combining semantic and keyword search with MMR and reranking

#     Parameters:
#     - query: User's search query
#     - faiss_index: FAISS index for semantic search
#     - metadata: Chunk metadata
#     - embedding_model: SentenceTransformer model for embeddings
#     - bm25_model: BM25 model for keyword search
#     - cross_encoder: CrossEncoder model for reranking
#     - k: Number of results to return
#     - lambda_param: Balance between semantic and keyword search (0-1)
#     - section_filter: Optional filter for specific sections
#     """
#     # Apply section filter if provided
#     if section_filter:
#         filtered_indices = [i for i, chunk in enumerate(metadata)
#                            if chunk.get('section_title') == section_filter or
#                               chunk.get('type') == section_filter]

#         if not filtered_indices:
#             print(f"No chunks found with section filter: {section_filter}")
#             return []
#     else:
#         filtered_indices = list(range(len(metadata)))

#     # Step 1: Semantic search with FAISS
#     query_embedding = embedding_model.encode([query])[0].reshape(1, -1).astype('float32')

#     semantic_k = min(k * 2, len(filtered_indices))  # Get more results initially for diversity
#     distances, indices = faiss_index.search(query_embedding, semantic_k)

#     # For L2 distance, smaller is better - need to convert to similarity scores
#     max_dist = np.max(distances[0]) + 1  # Add 1 to ensure positive values

#     # Filter results by section if needed
#     if section_filter:
#         semantic_scores = {}
#         for i, idx in enumerate(indices[0]):
#             if idx in filtered_indices:
#                 semantic_scores[idx] = 1 - (distances[0][i] / max_dist)  # Convert distance to similarity score
#     else:
#         semantic_scores = {idx: 1 - (distances[0][i] / max_dist) for i, idx in enumerate(indices[0])}

#     # Step 2: Keyword search with BM25
#     tokenized_query = word_tokenize(query.lower())
#     bm25_scores = bm25_model.get_scores(tokenized_query)

#     # Normalize BM25 scores
#     if max(bm25_scores) > 0:
#         bm25_scores = bm25_scores / max(bm25_scores)

#     # Create dictionary of BM25 scores
#     keyword_scores = {}
#     for i in filtered_indices:
#         keyword_scores[i] = bm25_scores[i]

#     # Step 3: Combine scores for hybrid search
#     hybrid_scores = {}
#     for idx in set(list(semantic_scores.keys()) + list(keyword_scores.keys())):
#         if idx in filtered_indices:
#             sem_score = semantic_scores.get(idx, 0)
#             key_score = keyword_scores.get(idx, 0)
#             hybrid_scores[idx] = lambda_param * sem_score + (1 - lambda_param) * key_score

#     # Get top k*2 results for MMR
#     sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]

#     # Step 4: Apply Maximum Marginal Relevance to reduce redundancy
#     mmr_results = mmr(sorted_results, metadata, embedding_model, k)

#     # Step 5: Rerank results if cross-encoder is provided
#     if cross_encoder:
#         reranked_results = rerank_results(query, mmr_results, metadata, cross_encoder)
#         results = reranked_results
#     else:
#         results = [(idx, score) for idx, score in mmr_results]

#     # Step 6: Add context to results
#     final_results = []
#     for idx, score in results[:k]:
#         chunk = metadata[idx]

#         # Add surrounding context
#         if chunk.get('type') == 'text' and chunk.get('chunk_id', 0) > 0 and chunk.get('chunk_id', 0) < len(metadata) - 1:
#             prev_chunk = metadata[chunk['chunk_id'] - 1]
#             next_chunk = metadata[chunk['chunk_id'] + 1]

#             context = {
#                 'text': chunk['text'],
#                 'metadata': chunk,
#                 'score': score,
#                 'prev_text': prev_chunk.get('text', '')[:200] + '...' if prev_chunk.get('type') == 'text' else '',
#                 'next_text': next_chunk.get('text', '')[:200] + '...' if next_chunk.get('type') == 'text' else ''
#             }
#         else:
#             context = {
#                 'text': chunk['text'],
#                 'metadata': chunk,
#                 'score': score,
#                 'prev_text': '',
#                 'next_text': ''
#             }

#         final_results.append(context)

#     return final_results

# def select_top_contexts(retrieved_results, num_contexts=2):
#     """
#     Select the top N most relevant contexts based on score
#     """
#     # Sort results by score in descending order
#     sorted_results = sorted(retrieved_results, key=lambda x: x['score'], reverse=True)
    
#     # Select top contexts, ensuring they have a positive relevance score
#     top_contexts = [
#         result for result in sorted_results 
#         if result['score'] > 0  # Only use contexts with positive relevance score
#     ][:num_contexts]
    
#     return top_contexts

# def format_retrieved_context(retrieved_results):
#     """
#     Format top 2 retrieved results into a coherent context string
#     """
#     # Select top 2 contexts
#     selected_contexts = select_top_contexts(retrieved_results)
    
#     context_parts = []
#     for i, result in enumerate(selected_contexts, 1):
#         # Include previous and next text if available
#         context_text = result['text']
#         if result.get('prev_text'):
#             context_text = f"[Previous context: {result['prev_text']}]\n\n{context_text}"
#         if result.get('next_text'):
#             context_text = f"{context_text}\n\n[Next context: {result['next_text']}]"
            
#         context_parts.append(f"Context {i} (Score: {result['score']:.2f}):\n{context_text}")
    
#     return "\n\n".join(context_parts)

# def generate_mistral_response(query, retrieved_context, api_key, company_name="NIKE"):
#     """
#     Generate a response using Mistral API with retrieved contexts
#     """
#     # Construct the prompt with contexts
#     prompt = f"""You are an expert assistant analyzing {company_name}'s 10-K report.
# Use ONLY the following two most relevant contexts to answer the query precisely and comprehensively.

# Query: {query}

# Top 2 Retrieved Contexts:
# {retrieved_context}

# Important Guidelines:
# 1. Base your response STRICTLY on the provided contexts
# 2. Do not introduce information not present in these contexts
# 3. If the context includes previous or next context indicators, incorporate that information appropriately
# 4. Provide a detailed and accurate response
# """

#     # Prepare the API request
#     headers = {
#         "Content-Type": "application/json",
#         "Accept": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
    
#     payload = {
#         "model": "mistral-medium",  # You can change to other models: mistral-tiny, mistral-small, mistral-large
#         "messages": [
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 1024
#     }
    
#     # Send request to Mistral API
#     try:
#         response = requests.post(
#             "https://api.mistral.ai/v1/chat/completions",
#             headers=headers,
#             data=json.dumps(payload)
#         )
        
#         # Check for successful response
#         if response.status_code == 200:
#             response_json = response.json()
#             return response_json["choices"][0]["message"]["content"]
#         else:
#             return f"Error from Mistral API: {response.status_code} - {response.text}"
            
#     except Exception as e:
#         return f"Error calling Mistral API: {str(e)}"

# def rag_pipeline(query, api_key, company_name='NIKE', lambda_param=0.5, section_filter=None):
#     """
#     Full RAG pipeline: retrieve, generate, and return response
#     """
#     # Load RAG components
#     index, metadata, embedding_model, bm25_model, cross_encoder = load_rag_components(company_name)
    
#     # Perform hybrid search
#     with st.spinner("Retrieving relevant information..."):
#         retrieved_results = hybrid_search(
#             query, index, metadata, embedding_model, bm25_model,
#             cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
#             section_filter=section_filter
#         )
    
#     # Format top 2 contexts
#     context = format_retrieved_context(retrieved_results)
    
#     # Generate response using Mistral API
#     with st.spinner("Generating response with Mistral API..."):
#         response = generate_mistral_response(query, context, api_key, company_name)
    
#     return {
#         'query': query,
#         'retrieved_results': retrieved_results,
#         'top_2_contexts': context.split('\n\n'),
#         'response': response
#     }

# def check_gpu():
#     if torch.cuda.is_available():
#         gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
#         memory_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
#         st.sidebar.success(f"✅ Using GPU for embeddings\n{gpu_info}\n{memory_info}")
#     else:
#         st.sidebar.warning("⚠️ GPU not detected. Using CPU for embeddings (may be slower)")

# def main():
#     st.title("10-K Report Analysis RAG System")
    
#     # Custom CSS for better styling
#     st.markdown("""
#     <style>
#     .stApp {
#         max-width: 1200px;
#         margin: 0 auto;
#     }
#     .result-container {
#         background-color: #f8f9fa;
#         padding: 20px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#         color: #333333;  /* Adding explicit dark text color */
#     }
#     </style>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#     This application allows you to query a company's 10-K report using a Retrieval-Augmented Generation (RAG) system.
#     The system retrieves relevant information from the report and generates a detailed response to your query.
#     """)
    
#     # Sidebar with company selection and advanced settings
#     st.sidebar.title("Settings")
#     company_name = st.sidebar.selectbox(
#         "Select company:",
#         ["NIKE", "APPLE", "MICROSOFT", "TESLA"],  # Add all available companies
#         index=0
#     )
    
#     # API Key input (securely)
#     api_key = st.sidebar.text_input("Enter Mistral API Key:", type="password")
    
#     # Advanced settings expander in sidebar
#     with st.sidebar.expander("Advanced Settings"):
#         lambda_param = st.slider(
#             "Hybrid search balance (λ):",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.5,
#             step=0.1,
#             help="Balance between semantic (1.0) and keyword (0.0) search"
#         )
        
#         section_filter = st.text_input(
#             "Section filter (optional):",
#             value="",
#             help="Filter results by section title or type (leave empty for all sections)"
#         )
        
#         mistral_model = st.selectbox(
#             "Mistral Model:",
#             ["mistral-medium", "mistral-small", "mistral-large", "mistral-tiny"],
#             index=0,
#             help="Select Mistral model to use (medium is recommended balance)"
#         )
        
#         if not section_filter:
#             section_filter = None
    
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Model Information")
#     st.sidebar.markdown(f"""
#     - **LLM**: Mistral API ({mistral_model})
#     - **Embedding Model**: all-mpnet-base-v2
#     - **Re-ranker**: ms-marco-MiniLM-L-6-v2
#     """)
    
#     # Main content
#     st.header(f"Query {company_name}'s 10-K Report")
    
#     # Example queries
#     st.markdown("### Example Queries")
#     example_queries = [
#         "What are the main revenue sources?",
#         "Describe the strategy for international markets",
#         "What are the key financial risks?",
#         "How does the company approach sustainability?",
#         "What are the major competitors mentioned?"
#     ]
    
#     # Create columns for example query buttons
#     cols = st.columns(3)
#     example_buttons = {}
    
#     for i, query in enumerate(example_queries):
#         col_idx = i % 3
#         with cols[col_idx]:
#             example_buttons[query] = st.button(query)
    
#     # Query input
#     query = st.text_area("Enter your query:", height=100, placeholder="Type your question about the company's 10-K report here...")
    
#     check_gpu()
    
#     # Set query if example button is clicked
#     for example_query, clicked in example_buttons.items():
#         if clicked:
#             query = example_query
#             # Need to rerun to update the text area
#             st.experimental_rerun()
    
#     # Process button
#     submit_col1, submit_col2 = st.columns([1, 5])
#     with submit_col1:
#         submit_button = st.button("Submit Query", type="primary")
    
#     if submit_button and query:
#         if not api_key:
#             st.error("Please enter your Mistral API key in the sidebar to continue.")
#         else:
#             start_time = time.time()
            
#             # Run RAG pipeline
#             result = rag_pipeline(
#                 query, 
#                 api_key,
#                 company_name=company_name,
#                 lambda_param=lambda_param,
#                 section_filter=section_filter
#             )
            
#             # Display processing time
#             processing_time = time.time() - start_time
#             st.info(f"Query processed in {processing_time:.2f} seconds")
            
#             st.markdown("## Response")
#             st.info(result["response"])  # Uses Streamlit's native info box with proper styling
            
#             # Show retrieved contexts in an expander
#             with st.expander("View retrieved contexts"):
#                 for i, context in enumerate(result['top_2_contexts']):
#                     st.markdown(f"### {context.split(':', 1)[0]}")
#                     parts = context.split(':\n', 1)  # Split on first occurrence only
#                     if len(parts) > 1:
#                         st.markdown(parts[1])
#                     else:
#                         st.markdown(parts[0])  # Just display the whole context if no split found
#                     st.markdown("---")
            
#             # Advanced results analysis
#             with st.expander("View all retrieved documents with context"):
#                 for i, result_item in enumerate(result['retrieved_results'], 1):
#                     col1, col2 = st.columns([1, 4])
                    
#                     with col1:
#                         st.markdown(f"**Score: {result_item['score']:.3f}**")
#                         if 'metadata' in result_item:
#                             metadata = result_item['metadata']
#                             if 'section_title' in metadata:
#                                 st.markdown(f"**Section: {metadata['section_title']}**")
#                             if 'page' in metadata:
#                                 st.markdown(f"**Page: {metadata['page']}**")
                    
#                     with col2:
#                         # Display previous context if available
#                         if result_item.get('prev_text'):
#                             st.markdown("**Previous Context:**")
#                             st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['prev_text']}</div>", 
#                                         unsafe_allow_html=True)
                        
#                         # Display main text
#                         st.markdown("**Main Text:**")
#                         st.markdown(result_item['text'])
                        
#                         # Display next context if available
#                         if result_item.get('next_text'):
#                             st.markdown("**Next Context:**")
#                             st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['next_text']}</div>", 
#                                         unsafe_allow_html=True)
                    
#                     st.markdown("---")

# if __name__ == "__main__":
#     main()





import streamlit as st
import torch
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
from nltk.tokenize import word_tokenize
import nltk
import os
import requests
import json
from bs4 import BeautifulSoup
import re
import pandas as pd
from typing import List, Dict, Tuple, Any
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
import tempfile
import os
import google.generativeai as genai



# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Download NLTK data for tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



# Store API key as a global variable
API_KEY = "grL3IacMpVhmjmfNW6ZNBismxLyNS7gV"
GEMINI_API_KEY = "AIzaSyAfxHwFSHC-5_h3BQ8GR0N1H6FAfyJqqUI"  # Add this line

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_rag_components(company_name, cross_encoder_model="ms-marco-MiniLM-L-6-v2"):
    """
    Load the pre-processed RAG components with caching for better performance
    Uses folder structure: Company 10-K's/{company_name}/{files}
    
    Parameters:
    - company_name: Name of the company whose 10-K report is being analyzed
    - cross_encoder_model: Name of the cross-encoder model to use for reranking
                           (default: 'ms-marco-MiniLM-L-6-v2')
    """
    base_path = f"Company 10-K's/{company_name}"
    
    with st.spinner(f"Loading RAG components for {company_name}..."):
        # Load FAISS index
        index = faiss.read_index(f"{base_path}/{company_name.lower()}_10k_enhanced_index.faiss")
        
        # Load metadata
        with open(f"{base_path}/{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        
        # Load BM25 model
        with open(f"{base_path}/{company_name.lower()}_10k_bm25.pkl", "rb") as f:
            bm25_model = pickle.load(f)
        
        # Load embedding model
        embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load Cross Encoder - Fix for duplicate prefix issue
        # Check if the model name already starts with 'cross-encoder/'
        if cross_encoder_model.startswith('cross-encoder/'):
            cross_encoder = CrossEncoder(cross_encoder_model)
        else:
            cross_encoder = CrossEncoder(f'cross-encoder/{cross_encoder_model}')
        
        return index, metadata, embedding_model, bm25_model, cross_encoder

def mmr(sorted_results, metadata, embedding_model, top_k, diversity=0.3):
    """
    Apply Maximum Marginal Relevance to reduce redundancy in results
    
    Parameters:
    - sorted_results: List of (index, score) tuples sorted by relevance
    - metadata: Document metadata
    - embedding_model: SentenceTransformer model
    - top_k: Number of results to return
    - diversity: Balance between relevance and diversity (0-1)
    
    Returns:
    - Reranked list of (index, score) tuples
    """
    if not sorted_results:
        return []
    
    # Get embeddings for all results
    indices = [idx for idx, _ in sorted_results]
    texts = [metadata[idx]['text'] for idx in indices]
    embeddings = embedding_model.encode(texts)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    # Select first result
    selected = [sorted_results[0]]
    selected_embeddings = [embeddings[0]]
    remaining = sorted_results[1:]
    remaining_embeddings = embeddings[1:]
    
    # Select remaining results using MMR
    while len(selected) < top_k and remaining:
        # Calculate similarity to already selected documents
        similarities = np.dot(remaining_embeddings, np.array(selected_embeddings).T)
        max_similarities = np.max(similarities, axis=1)
        
        # Calculate MMR scores
        mmr_scores = [(1 - diversity) * score - diversity * sim 
                      for (_, score), sim in zip(remaining, max_similarities)]
        
        # Select document with highest MMR score
        next_idx = np.argmax(mmr_scores)
        selected.append(remaining[next_idx])
        selected_embeddings.append(remaining_embeddings[next_idx])
        
        # Remove selected document
        remaining.pop(next_idx)
        remaining_embeddings = np.delete(remaining_embeddings, next_idx, axis=0)
    
    return selected

def rerank_results(query, results, metadata, cross_encoder):
    """
    Rerank results using a cross-encoder model
    
    Parameters:
    - query: User query
    - results: List of (index, score) tuples
    - metadata: Document metadata
    - cross_encoder: CrossEncoder model
    
    Returns:
    - Reranked list of (index, score) tuples
    """
    if not results:
        return []
    
    # Prepare pairs for cross-encoder
    pairs = [[query, metadata[idx]['text']] for idx, _ in results]
    
    # Get cross-encoder scores
    cross_scores = cross_encoder.predict(pairs)
    
    # Create new results with updated scores
    reranked = [(idx, float(score)) for (idx, _), score in zip(results, cross_scores)]
    
    # Sort by cross-encoder score
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    return reranked

def hybrid_search(query, faiss_index, metadata, embedding_model, bm25_model,
                 cross_encoder=None, k=10, lambda_param=0.5, section_filter=None):
    """
    Perform hybrid search combining semantic and keyword search with MMR and reranking

    Parameters:
    - query: User's search query
    - faiss_index: FAISS index for semantic search
    - metadata: Chunk metadata
    - embedding_model: SentenceTransformer model for embeddings
    - bm25_model: BM25 model for keyword search
    - cross_encoder: CrossEncoder model for reranking
    - k: Number of results to return
    - lambda_param: Balance between semantic and keyword search (0-1)
    - section_filter: Optional filter for specific sections
    """
    # Apply section filter if provided
    if section_filter:
        filtered_indices = [i for i, chunk in enumerate(metadata)
                           if chunk.get('section_title') == section_filter or
                              chunk.get('type') == section_filter]

        if not filtered_indices:
            print(f"No chunks found with section filter: {section_filter}")
            return []
    else:
        filtered_indices = list(range(len(metadata)))

    # Step 1: Semantic search with FAISS
    query_embedding = embedding_model.encode([query])[0].reshape(1, -1).astype('float32')

    semantic_k = min(k * 2, len(filtered_indices))  # Get more results initially for diversity
    distances, indices = faiss_index.search(query_embedding, semantic_k)

    # For L2 distance, smaller is better - need to convert to similarity scores
    max_dist = np.max(distances[0]) + 1  # Add 1 to ensure positive values

    # Filter results by section if needed
    if section_filter:
        semantic_scores = {}
        for i, idx in enumerate(indices[0]):
            if idx in filtered_indices:
                semantic_scores[idx] = 1 - (distances[0][i] / max_dist)  # Convert distance to similarity score
    else:
        semantic_scores = {idx: 1 - (distances[0][i] / max_dist) for i, idx in enumerate(indices[0])}

    # Step 2: Keyword search with BM25
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25_model.get_scores(tokenized_query)

    # Normalize BM25 scores
    if max(bm25_scores) > 0:
        bm25_scores = bm25_scores / max(bm25_scores)

    # Create dictionary of BM25 scores
    keyword_scores = {}
    for i in filtered_indices:
        keyword_scores[i] = bm25_scores[i]

    # Step 3: Combine scores for hybrid search
    hybrid_scores = {}
    for idx in set(list(semantic_scores.keys()) + list(keyword_scores.keys())):
        if idx in filtered_indices:
            sem_score = semantic_scores.get(idx, 0)
            key_score = keyword_scores.get(idx, 0)
            hybrid_scores[idx] = lambda_param * sem_score + (1 - lambda_param) * key_score

    # Get top k*2 results for MMR
    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]

    # Step 4: Apply Maximum Marginal Relevance to reduce redundancy
    mmr_results = mmr(sorted_results, metadata, embedding_model, k)

    # Step 5: Rerank results if cross-encoder is provided
    if cross_encoder:
        reranked_results = rerank_results(query, mmr_results, metadata, cross_encoder)
        results = reranked_results
    else:
        results = [(idx, score) for idx, score in mmr_results]

    # Step 6: Add context to results
    final_results = []
    for idx, score in results[:k]:
        chunk = metadata[idx]

        # Add surrounding context
        if chunk.get('type') == 'text' and chunk.get('chunk_id', 0) > 0 and chunk.get('chunk_id', 0) < len(metadata) - 1:
            prev_chunk = metadata[chunk['chunk_id'] - 1]
            next_chunk = metadata[chunk['chunk_id'] + 1]

            context = {
                'text': chunk['text'],
                'metadata': chunk,
                'score': score,
                'prev_text': prev_chunk.get('text', '')[:200] + '...' if prev_chunk.get('type') == 'text' else '',
                'next_text': next_chunk.get('text', '')[:200] + '...' if next_chunk.get('type') == 'text' else ''
            }
        else:
            context = {
                'text': chunk['text'],
                'metadata': chunk,
                'score': score,
                'prev_text': '',
                'next_text': ''
            }

        final_results.append(context)

    return final_results

def select_top_contexts(retrieved_results, num_contexts=2):
    # Sort results by score in descending order (highest scores first)
    sorted_results = sorted(retrieved_results, key=lambda x: x['score'], reverse=True)
    
    # Debug print
    print(f"Scores of all results: {[result['score'] for result in sorted_results]}")
    
    # Select top contexts regardless of score value
    top_contexts = sorted_results[:num_contexts]
    
    print(f"Selected {len(top_contexts)} contexts")
    
    return top_contexts

def format_retrieved_context(retrieved_results):
    """
    Format top 2 retrieved results into a coherent context string
    """
    # Select top 2 contexts
    selected_contexts = select_top_contexts(retrieved_results)
    
    context_parts = []
    for i, result in enumerate(selected_contexts, 1):
        # Include previous and next text if available
        context_text = result['text']
        if result.get('prev_text'):
            context_text = f"[Previous context: {result['prev_text']}]\n\n{context_text}"
        if result.get('next_text'):
            context_text = f"{context_text}\n\n[Next context: {result['next_text']}]"
            
        context_parts.append(f"Context {i} (Score: {result['score']:.2f}):\n{context_text}")
    
    return "\n\n".join(context_parts)

def generate_mistral_response(query, retrieved_context, api_key, company_name="NIKE", model_name="mistral-medium"):
    """
    Generate a response using Mistral API with retrieved contexts
    """
    # Construct the prompt with contexts but instruct not to mention context numbers
    prompt = f"""You are an expert assistant analyzing {company_name}'s 10-K report.
Use the following contexts to answer the query precisely and comprehensively IN ABOUT 100 WORDS.

Query: {query}

Retrieved Contexts:
{retrieved_context}

Important Guidelines:
1. Base your response STRICTLY on the provided contexts
2. Do not introduce information not present in these contexts
3. If the context includes previous or next context indicators, incorporate that information appropriately
4. Provide a detailed and accurate response IN ABOUT 100 WORDS
5. DO NOT explicitly mention "Context 1" or "Context 2" in your response - just provide a coherent answer without referencing the source numbering
6. Present the information as a unified, seamless response
"""

    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model_name,  # Use the selected model from parameters
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # Send request to Mistral API
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        
        # Check for successful response
        if response.status_code == 200:
            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
        else:
            return f"Error from Mistral API: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling Mistral API: {str(e)}"

def generate_gemini_response(query, retrieved_context, company_name="NIKE", model_name="gemini-2.0-flash"):
    """
    Generate a response using Google's Gemini API with retrieved contexts
    """
    # Restructure the prompt to make it clearer that contexts are provided
    prompt = f"""You are an expert assistant analyzing {company_name}'s 10-K report.
I will provide you with relevant contexts from the report, and you should use ONLY this information to answer the query IN ABOUT 100 WORDS.

Query: {query}

Here are the relevant contexts from the {company_name} 10-K report:
{retrieved_context}

Important Guidelines:
1. Your answer must be based SOLELY on the contexts provided above
2. Do not introduce information not present in these contexts
3. If the context includes previous or next context indicators, incorporate that information appropriately
4. Provide a detailed and accurate response IN ABOUT 100 WORDS
5. DO NOT mention "Context 1" or "Context 2" in your response - craft a coherent answer without referencing the source numbering
6. Present the information as a unified response based only on the {company_name} 10-K data provided
"""

    try:
        # Initialize the Gemini model based on the chosen model name
        if model_name == "gemini-2.0-flash":
            model = genai.GenerativeModel('gemini-2.0-flash')
        elif model_name == "gemini-pro-vision":
            model = genai.GenerativeModel('gemini-2.0-flash')
        else:  # Default to gemini-pro
            model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Set structured parameters to control generation
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more factual responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Generate response with explicit configuration
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Return the text response
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

def rag_pipeline(query, company_name='NIKE', lambda_param=0.5, section_filter=None, llm_provider="mistral", model_name="mistral-medium", cross_encoder_model="ms-marco-MiniLM-L-6-v2"):
    """
    Full RAG pipeline: retrieve, generate, and return response
    Now supports both Mistral and Gemini, and different cross-encoder models
    
    Parameters:
    - query: User's search query
    - company_name: Name of the company whose 10-K report is being analyzed
    - lambda_param: Balance between semantic and keyword search (0-1)
    - section_filter: Optional filter for specific sections
    - llm_provider: LLM provider to use (mistral or gemini)
    - model_name: Model name for the selected LLM provider
    - cross_encoder_model: Cross-encoder model to use for reranking
    """
    # Load RAG components with specified cross-encoder model
    index, metadata, embedding_model, bm25_model, cross_encoder = load_rag_components(company_name, cross_encoder_model)
    
    # Perform hybrid search
    with st.spinner("Retrieving relevant information..."):
        retrieved_results = hybrid_search(
            query, index, metadata, embedding_model, bm25_model,
            cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
            section_filter=section_filter
        )
    
    # Format top 2 contexts
    context = format_retrieved_context(retrieved_results)
    
    # Generate response using selected LLM
    with st.spinner(f"Decoding financial insights using {llm_provider.capitalize()}, please hold on....."):
        if llm_provider == "mistral":
            response = generate_mistral_response(query, context, API_KEY, company_name, model_name)
        elif llm_provider == "gemini":
            response = generate_gemini_response(query, context, company_name, model_name)
        else:
            response = "Error: Invalid LLM provider selected."
    
    return {
        'query': query,
        'retrieved_results': retrieved_results,
        'top_2_contexts': context.split('\n\n'),
        'response': response
    }

def check_gpu():
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        memory_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        st.sidebar.success(f"✅ Using GPU for embeddings\n{gpu_info}\n{memory_info}")
    else:
        st.sidebar.warning("⚠️ GPU not detected. Using CPU for embeddings (may be slower)")




def similar(a, b):
    """Calculate the similarity ratio between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_ticker_from_company_name(company_name, csv_path="company_tickers_RAG.csv", threshold=0.75):
    """
    Find the most similar company name in the CSV file and return its ticker
    
    Parameters:
    - company_name: Company name from dropdown
    - csv_path: Path to CSV file with company names and tickers
    - threshold: Minimum similarity score to consider a match
    
    Returns:
    - ticker: Ticker symbol or None if no match found
    """
    try:
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Find the best match
        best_match = None
        best_score = 0
        
        for index, row in df.iterrows():
            csv_company = row.get('Company', '')
            score = similar(company_name, csv_company)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = row.get('Ticker', '')
        
        return best_match
    except Exception as e:
        st.warning(f"Error finding ticker: {str(e)}")
        return None


def get_company_info(ticker_symbol):
    """
    Get company information from Yahoo Finance
    
    Parameters:
    - ticker_symbol: Company's ticker symbol
    
    Returns:
    - Dictionary with company information
    """
    if not ticker_symbol:
        return None
    
    try:
        # Get ticker info
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Structure the data
        company_data = {
            "basic_info": {
                "name": info.get('shortName', info.get('longName', 'N/A')),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "headquarters": f"{info.get('city', 'N/A')}, {info.get('state', '')}, {info.get('country', '')}".replace(", , ", ", ").strip(", ")
            },
            "stock_info": {
                "symbol": ticker_symbol,
                "current_price": info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                "open_price": info.get('open', info.get('regularMarketOpen', 'N/A')),
                "previous_close": info.get('previousClose', info.get('regularMarketPreviousClose', 'N/A')),
                "day_high": info.get('dayHigh', info.get('regularMarketDayHigh', 'N/A')),
                "day_low": info.get('dayLow', info.get('regularMarketDayLow', 'N/A')),
                "market_cap": info.get('marketCap', 'N/A'),
                "volume": info.get('volume', info.get('regularMarketVolume', 'N/A'))
            },
            "valuation_metrics": {
                "pe_ratio": info.get('trailingPE', info.get('forwardPE', 'N/A')),
                "pb_ratio": info.get('priceToBook', 'N/A'),
                "roe": info.get('returnOnEquity', 'N/A'),
                "enterprise_value": info.get('enterpriseValue', 'N/A'),
                "ebitda": info.get('ebitda', 'N/A')
            },
            "market_sentiment": {
                "recommendation": info.get('recommendationKey', 'N/A'),
                "target_price": info.get('targetMeanPrice', 'N/A'),
                "target_high": info.get('targetHighPrice', 'N/A'),
                "target_low": info.get('targetLowPrice', 'N/A'),
                "analyst_count": info.get('numberOfAnalystOpinions', 'N/A')
            },
            "company_profile": {
                "business_summary": info.get('longBusinessSummary', 'N/A'),
                "website": info.get('website', 'N/A'),
                "employees": info.get('fullTimeEmployees', 'N/A')
            },
            "key_executives": [],
            "financials": {
                "income_statement": {},
                "balance_sheet": {},
                "cash_flow": {}
            },
            "news": []
        }
        
        # Try to get executives information
        try:
            if 'companyOfficers' in info and info['companyOfficers']:
                for officer in info['companyOfficers']:
                    executive = {
                        "name": officer.get('name', 'N/A'),
                        "title": officer.get('title', 'N/A'),
                        "age": officer.get('age', 'N/A'),
                        "salary": officer.get('totalPay', 'N/A')
                    }
                    company_data["key_executives"].append(executive)
        except:
            pass  # Skip executives if data is unavailable
        
        # Get financial statements
        try:
            # Income Statement
            income_stmt = ticker.income_stmt
            if not income_stmt.empty:
                company_data["financials"]["income_statement"] = {
                    "total_revenue": income_stmt.loc["Total Revenue"].iloc[0] if "Total Revenue" in income_stmt.index else 'N/A',
                    "gross_profit": income_stmt.loc["Gross Profit"].iloc[0] if "Gross Profit" in income_stmt.index else 'N/A',
                    "operating_income": income_stmt.loc["Operating Income"].iloc[0] if "Operating Income" in income_stmt.index else 'N/A',
                    "net_income": income_stmt.loc["Net Income"].iloc[0] if "Net Income" in income_stmt.index else 'N/A',
                    "eps": income_stmt.loc["Basic EPS"].iloc[0] if "Basic EPS" in income_stmt.index else 'N/A'
                }
            
            # Balance Sheet
            balance_sheet = ticker.balance_sheet
            if not balance_sheet.empty:
                company_data["financials"]["balance_sheet"] = {
                    "total_assets": balance_sheet.loc["Total Assets"].iloc[0] if "Total Assets" in balance_sheet.index else 'N/A',
                    "total_liabilities": balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0] if "Total Liabilities Net Minority Interest" in balance_sheet.index else 'N/A',
                    "total_equity": balance_sheet.loc["Total Equity Gross Minority Interest"].iloc[0] if "Total Equity Gross Minority Interest" in balance_sheet.index else 'N/A',
                    "cash_and_equivalents": balance_sheet.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in balance_sheet.index else 'N/A',
                    "total_debt": balance_sheet.loc["Total Debt"].iloc[0] if "Total Debt" in balance_sheet.index else 'N/A'
                }
            
            # Cash Flow
            cash_flow = ticker.cashflow
            if not cash_flow.empty:
                company_data["financials"]["cash_flow"] = {
                    "operating_cash_flow": cash_flow.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cash_flow.index else 'N/A',
                    "investing_cash_flow": cash_flow.loc["Investing Cash Flow"].iloc[0] if "Investing Cash Flow" in cash_flow.index else 'N/A',
                    "financing_cash_flow": cash_flow.loc["Financing Cash Flow"].iloc[0] if "Financing Cash Flow" in cash_flow.index else 'N/A',
                    "free_cash_flow": cash_flow.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cash_flow.index else 'N/A',
                    "capital_expenditure": cash_flow.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cash_flow.index else 'N/A'
                }
        except Exception as e:
            st.warning(f"Could not retrieve financial statements: {str(e)}")
            pass
        
        # Get latest news
        try:
            news_items = ticker.news
            if news_items:
                for i, news in enumerate(news_items[:2]):  # Get only 2 latest news
                    company_data["news"].append({
                        "title": news.get('title', 'N/A'),
                        "publisher": news.get('publisher', 'N/A'),
                        "link": news.get('link', '#'),
                        "published": news.get('providerPublishTime', 'N/A')
                    })
        except:
            pass
        
        return company_data
    
    except Exception as e:
        st.warning(f"Error retrieving company data: {str(e)}")
        return None

def display_company_info(company_name):
    """
    Display company information in the Streamlit app
    
    Parameters:
    - company_name: Selected company name
    """
    # Get ticker symbol
    ticker = get_ticker_from_company_name(company_name)
    
    if not ticker:
        st.warning(f"Could not find ticker symbol for {company_name}")
        return
    
    # Get company information
    company_data = get_company_info(ticker)
    
    if not company_data:
        st.warning(f"Could not retrieve information for {company_name} ({ticker})")
        return
    
    # Display company basic info
    basic_info = company_data["basic_info"]

    st.markdown(
        f"""
        <div style="
            background-color: rgba(240, 242, 246, 0.9); /* Light gray with slight transparency */
            padding: 20px;
            border-radius: 30px; /* Increased for smoother edges */
            text-align: center;
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1); /* Optional: Adds subtle shadow for depth */
        ">
            <div style="font-size: 24px; font-weight: bold; color: black; text-align: center;">
                {basic_info['name']} ({company_data['stock_info']['symbol']})
            </div>
            <div style="font-size: 16px; color: black; text-align: center;">
                <strong>Sector:</strong> {basic_info['sector']} | <strong>Industry:</strong> {basic_info['industry']} | <strong>HQ:</strong> {basic_info['headquarters']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
)

        
    st.markdown("***")

    # Expandable section for detailed information
    with st.expander("View Detailed Company Information", expanded=False):
        # Create tabs for different categories of information
        overview_tab, financials_tab, news_tab = st.tabs(["Overview", "Financial Statements", "Latest News"])
        
        with overview_tab:
            # Stock information in columns
            st.subheader("Market Data")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${company_data['stock_info']['current_price']}" if isinstance(company_data['stock_info']['current_price'], (int, float)) else company_data['stock_info']['current_price'])
                st.metric("Market Cap", f"${company_data['stock_info']['market_cap']:,}" if isinstance(company_data['stock_info']['market_cap'], (int, float)) else company_data['stock_info']['market_cap'])
            
            with col2:
                st.metric("Open", f"${company_data['stock_info']['open_price']}" if isinstance(company_data['stock_info']['open_price'], (int, float)) else company_data['stock_info']['open_price'])
                st.metric("Volume", f"{company_data['stock_info']['volume']:,}" if isinstance(company_data['stock_info']['volume'], (int, float)) else company_data['stock_info']['volume'])
            
            with col3:
                st.metric("Day High", f"${company_data['stock_info']['day_high']}" if isinstance(company_data['stock_info']['day_high'], (int, float)) else company_data['stock_info']['day_high'])
                st.metric("Day Low", f"${company_data['stock_info']['day_low']}" if isinstance(company_data['stock_info']['day_low'], (int, float)) else company_data['stock_info']['day_low'])
            
            with col4:
                st.metric("Previous Close", f"${company_data['stock_info']['previous_close']}" if isinstance(company_data['stock_info']['previous_close'], (int, float)) else company_data['stock_info']['previous_close'])
                st.metric("Employees", f"{company_data['company_profile']['employees']:,}" if isinstance(company_data['company_profile']['employees'], (int, float)) else company_data['company_profile']['employees'])
            
            # Valuation metrics
            st.subheader("Valuation Metrics")
            val_col1, val_col2, val_col3, val_col4, val_col5 = st.columns(5)
            
            with val_col1:
                st.metric("P/E Ratio", f"{company_data['valuation_metrics']['pe_ratio']:.2f}" if isinstance(company_data['valuation_metrics']['pe_ratio'], (int, float)) else company_data['valuation_metrics']['pe_ratio'])
            
            with val_col2:
                st.metric("P/B Ratio", f"{company_data['valuation_metrics']['pb_ratio']:.2f}" if isinstance(company_data['valuation_metrics']['pb_ratio'], (int, float)) else company_data['valuation_metrics']['pb_ratio'])
            
            with val_col3:
                st.metric("ROE", f"{company_data['valuation_metrics']['roe'] * 100:.2f}%" if isinstance(company_data['valuation_metrics']['roe'], (int, float)) else company_data['valuation_metrics']['roe'])
            
            with val_col4:
                st.metric("Enterprise Value", f"${company_data['valuation_metrics']['enterprise_value']:,}" if isinstance(company_data['valuation_metrics']['enterprise_value'], (int, float)) else company_data['valuation_metrics']['enterprise_value'])
            
            with val_col5:
                st.metric("EBITDA", f"${company_data['valuation_metrics']['ebitda']:,}" if isinstance(company_data['valuation_metrics']['ebitda'], (int, float)) else company_data['valuation_metrics']['ebitda'])
            
            # Market sentiment
            st.subheader("Market Sentiment")
            sent_col1, sent_col2, sent_col3 = st.columns(3)
            
            with sent_col1:
                st.metric("Analyst Recommendation", company_data['market_sentiment']['recommendation'].capitalize() if company_data['market_sentiment']['recommendation'] != 'N/A' else 'N/A')
            
            with sent_col2:
                st.metric("Target Price", f"${company_data['market_sentiment']['target_price']:.2f}" if isinstance(company_data['market_sentiment']['target_price'], (int, float)) else company_data['market_sentiment']['target_price'])
            
            with sent_col3:
                st.metric("Analyst Count", company_data['market_sentiment']['analyst_count'] if company_data['market_sentiment']['analyst_count'] != 'N/A' else 'N/A')
            
            # Company description
            st.subheader("Business Summary")
            st.write(company_data['company_profile']['business_summary'])
            
            # Key executives
            if company_data["key_executives"]:
                st.subheader("Key Executives")
                exec_cols = st.columns(min(3, len(company_data["key_executives"])))
                
                for i, executive in enumerate(company_data["key_executives"]):
                    col_idx = i % len(exec_cols)
                    with exec_cols[col_idx]:
                        st.markdown(f"""
                        **{executive['name']}**  
                        {executive['title']}  
                        {"Age: " + str(executive['age']) if executive['age'] != 'N/A' else ""}  
                        {"Salary: $" + f"{executive['salary']:,.2f}" if isinstance(executive['salary'], (int, float)) else ""}
                        """)
        
        with financials_tab:
            # Create subtabs for different financial statements
            income_tab, balance_tab, cash_tab = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
            
            with income_tab:
                st.subheader("Income Statement Highlights")
                income_data = company_data["financials"]["income_statement"]
                
                if income_data:
                    income_col1, income_col2 = st.columns(2)
                    
                    with income_col1:
                        st.metric("Total Revenue", f"${income_data['total_revenue']:,}" if isinstance(income_data['total_revenue'], (int, float)) else income_data['total_revenue'])
                        st.metric("Gross Profit", f"${income_data['gross_profit']:,}" if isinstance(income_data['gross_profit'], (int, float)) else income_data['gross_profit'])
                        st.metric("Operating Income", f"${income_data['operating_income']:,}" if isinstance(income_data['operating_income'], (int, float)) else income_data['operating_income'])
                    
                    with income_col2:
                        st.metric("Net Income", f"${income_data['net_income']:,}" if isinstance(income_data['net_income'], (int, float)) else income_data['net_income'])
                        st.metric("EPS", f"${income_data['eps']}" if isinstance(income_data['eps'], (int, float)) else income_data['eps'])
                else:
                    st.info("Income statement data not available")
            
            with balance_tab:
                st.subheader("Balance Sheet Highlights")
                balance_data = company_data["financials"]["balance_sheet"]
                
                if balance_data:
                    balance_col1, balance_col2 = st.columns(2)
                    
                    with balance_col1:
                        st.metric("Total Assets", f"${balance_data['total_assets']:,}" if isinstance(balance_data['total_assets'], (int, float)) else balance_data['total_assets'])
                        st.metric("Total Liabilities", f"${balance_data['total_liabilities']:,}" if isinstance(balance_data['total_liabilities'], (int, float)) else balance_data['total_liabilities'])
                        st.metric("Total Equity", f"${balance_data['total_equity']:,}" if isinstance(balance_data['total_equity'], (int, float)) else balance_data['total_equity'])
                    
                    with balance_col2:
                        st.metric("Cash & Equivalents", f"${balance_data['cash_and_equivalents']:,}" if isinstance(balance_data['cash_and_equivalents'], (int, float)) else balance_data['cash_and_equivalents'])
                        st.metric("Total Debt", f"${balance_data['total_debt']:,}" if isinstance(balance_data['total_debt'], (int, float)) else balance_data['total_debt'])
                else:
                    st.info("Balance sheet data not available")
            
            with cash_tab:
                st.subheader("Cash Flow Highlights")
                cash_data = company_data["financials"]["cash_flow"]
                
                if cash_data:
                    cash_col1, cash_col2 = st.columns(2)
                    
                    with cash_col1:
                        st.metric("Operating Cash Flow", f"${cash_data['operating_cash_flow']:,}" if isinstance(cash_data['operating_cash_flow'], (int, float)) else cash_data['operating_cash_flow'])
                        st.metric("Investing Cash Flow", f"${cash_data['investing_cash_flow']:,}" if isinstance(cash_data['investing_cash_flow'], (int, float)) else cash_data['investing_cash_flow'])
                        st.metric("Financing Cash Flow", f"${cash_data['financing_cash_flow']:,}" if isinstance(cash_data['financing_cash_flow'], (int, float)) else cash_data['financing_cash_flow'])
                    
                    with cash_col2:
                        st.metric("Free Cash Flow", f"${cash_data['free_cash_flow']:,}" if isinstance(cash_data['free_cash_flow'], (int, float)) else cash_data['free_cash_flow'])
                        st.metric("Capital Expenditure", f"${cash_data['capital_expenditure']:,}" if isinstance(cash_data['capital_expenditure'], (int, float)) else cash_data['capital_expenditure'])
                else:
                    st.info("Cash flow data not available")
        
        # For news display in the news_tab section:
        with news_tab:
            st.subheader("Latest News")
            
            if company_data["news"]:
                for news_item in company_data["news"]:
                    published_date = 'N/A'
                    if isinstance(news_item['published'], (int, float)):
                        try:
                            published_date = datetime.fromtimestamp(news_item['published']).strftime('%Y-%m-%d %H:%M')
                        except:
                            published_date = 'N/A'
                            
                    st.markdown(f"""
                    ### {news_item['title']}
                    **Publisher:** {news_item['publisher']}  
                    **Published:** {published_date}  
                    [Read more]({news_item['link']})
                    ---
                    """)
            else:
                st.info("No recent news available")



def langchain_hybrid_search(query, faiss_index, metadata, embedding_model, bm25_model,
                           cross_encoder=None, k=10, lambda_param=0.5, section_filter=None):
    """
    Alternative implementation using Langchain components for hybrid search
    
    Parameters:
    - Same as hybrid_search function
    
    Returns:
    - Same format as hybrid_search
    """
    # Apply section filter if provided
    if section_filter:
        filtered_indices = [i for i, chunk in enumerate(metadata)
                           if chunk.get('section_title') == section_filter or
                              chunk.get('type') == section_filter]

        if not filtered_indices:
            print(f"No chunks found with section filter: {section_filter}")
            return []
    else:
        filtered_indices = list(range(len(metadata)))
    
    # Convert metadata to Langchain documents for BM25
    langchain_docs = []
    for idx in filtered_indices:
        langchain_docs.append(Document(
            page_content=metadata[idx]['text'],
            metadata={k: v for k, v in metadata[idx].items() if k != 'text'}
        ))
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = k
    
    # Create a temporary FAISS index using Langchain
    # We'll use the existing FAISS index for actual retrieval
    with tempfile.TemporaryDirectory() as temp_dir:
        # Wrap the embedding model
        lc_embeddings = HuggingFaceEmbeddings(
            model_name='all-mpnet-base-v2',
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Get documents from BM25
        bm25_results = bm25_retriever.get_relevant_documents(query)
        bm25_indices = [filtered_indices[langchain_docs.index(doc)] for doc in bm25_results if doc in langchain_docs]
        bm25_scores = [1.0 - (i / len(bm25_results)) for i in range(len(bm25_results))]  # Normalize scores
        
        # Get documents from FAISS (using original index)
        query_embedding = embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        semantic_k = min(k * 2, len(filtered_indices))
        distances, indices = faiss_index.search(query_embedding, semantic_k)
        
        # Convert to similarity scores (1 - normalized distance)
        max_dist = np.max(distances[0]) + 1  # Add 1 to ensure positive values
        semantic_indices = [int(idx) for idx in indices[0]]
        semantic_scores = [1 - (distances[0][i] / max_dist) for i in range(len(semantic_indices))]
        
        # Combine results (similar to your hybrid search)
        all_indices = list(set(bm25_indices + semantic_indices))
        
        # Calculate hybrid scores
        hybrid_scores = {}
        for i, idx in enumerate(all_indices):
            sem_score = 0
            if idx in semantic_indices:
                sem_idx = semantic_indices.index(idx)
                sem_score = semantic_scores[sem_idx]
            
            key_score = 0
            if idx in bm25_indices:
                key_idx = bm25_indices.index(idx)
                key_score = bm25_scores[key_idx]
            
            hybrid_scores[idx] = lambda_param * sem_score + (1 - lambda_param) * key_score
        
        # Sort by score
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k*2]
        
        # Apply MMR (using your existing function)
        mmr_results = mmr(sorted_results, metadata, embedding_model, k)
        
        # Rerank if cross-encoder is provided
        if cross_encoder:
            reranked_results = rerank_results(query, mmr_results, metadata, cross_encoder)
            results = reranked_results
        else:
            results = [(idx, score) for idx, score in mmr_results]
        
        # Format results like your hybrid_search function
        final_results = []
        for idx, score in results[:k]:
            chunk = metadata[idx]
            
            # Add surrounding context
            if chunk.get('type') == 'text' and chunk.get('chunk_id', 0) > 0 and chunk.get('chunk_id', 0) < len(metadata) - 1:
                prev_chunk = metadata[chunk['chunk_id'] - 1]
                next_chunk = metadata[chunk['chunk_id'] + 1]
                
                context = {
                    'text': chunk['text'],
                    'metadata': chunk,
                    'score': score,
                    'prev_text': prev_chunk.get('text', '')[:200] + '...' if prev_chunk.get('type') == 'text' else '',
                    'next_text': next_chunk.get('text', '')[:200] + '...' if next_chunk.get('type') == 'text' else ''
                }
            else:
                context = {
                    'text': chunk['text'],
                    'metadata': chunk,
                    'score': score,
                    'prev_text': '',
                    'next_text': ''
                }
                
            final_results.append(context)
        
        return final_results

def langchain_rag_pipeline(query, company_name='NIKE', lambda_param=0.5, section_filter=None, llm_provider="mistral", model_name="mistral-medium", cross_encoder_model="ms-marco-MiniLM-L-6-v2"):
    """
    Alternative RAG pipeline using Langchain components
    Now supports both Mistral and Gemini, and different cross-encoder models
    """
    # Load RAG components with the specified cross-encoder model
    index, metadata, embedding_model, bm25_model, cross_encoder = load_rag_components(company_name, cross_encoder_model)
    
    # Perform hybrid search using Langchain
    with st.spinner(f"Retrieving relevant information using Langchain..."):
        retrieved_results = langchain_hybrid_search(
            query, index, metadata, embedding_model, bm25_model,
            cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
            section_filter=section_filter
        )
    
    # Format top 2 contexts
    context = format_retrieved_context(retrieved_results)
    
    # Generate response using selected LLM
    with st.spinner(f"Decoding financial insights using {llm_provider.capitalize()}, please hold on....."):
        if llm_provider == "mistral":
            response = generate_mistral_response(query, context, API_KEY, company_name, model_name)
        elif llm_provider == "gemini":
            response = generate_gemini_response(query, context, company_name, model_name)
        else:
            response = "Error: Invalid LLM provider selected."
    
    return {
        'query': query,
        'retrieved_results': retrieved_results,
        'top_2_contexts': context.split('\n\n'),
        'response': response
    }

def main():
    st.markdown("<h3 style='text-align: center;'>10-K Report Analysis</h3>", unsafe_allow_html=True)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #333333;  /* Adding explicit dark text color */
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application allows you to query a company's 10-K report using a Retrieval-Augmented Generation (RAG) system.
    The system retrieves relevant information from the report and generates a detailed response to your query.
    """)
    
    # Sidebar with company selection and advanced settings
    st.sidebar.title("Settings")
    company_name = st.sidebar.selectbox(
        "Select company:",
        ["NIKE", "APPLE", "AMAZON", "BERKSHIRE HATHAWAY", "WALMART", "UNITEDHEALTH", 
        "CVS", "EXXON MOBIL", "Alphabet", "MCKESSON", "Cencora", "COSTCO", 
        "MICROSOFT", "AT&T", "CARDINAL HEALTH", "CHEVRON", "HOME DEPOT", 
        "Walgreens", "Marathon Petroleum", "Elevance Health", "FORD", 
        "General Motors", "JPMORGAN", "VERIZON", "Phillips", "VALERO ENERGY", 
        "Fannie Mae", "Dell", "METLIFE", "COMCAST", "PEPSICO", "INTEL", 
        "PROCTER & GAMBLE", "IBM", "CATERPILLAR", "PFIZER", "LOWES", 
        "LOCKHEED MARTIN", "GOLDMAN SACHS", "MORGAN STANLEY", "Tesla", 
        "CISCO", "JOHNSON & JOHNSON", "ORACLE", "Merck & Co.", "WELLS FARGO", 
        "HONEYWELL", "CITIGROUP", "RTX", "AbbVie", "NIKE", "APPLE"],
        index=0
    )
    
    # Display Yahoo Finance company information
    display_company_info(company_name)
    
    # Add radio button for selecting framework instead of checkbox
    st.sidebar.markdown("---")
    rag_framework = st.sidebar.radio(
        "Select RAG framework:",
        ["Custom RAG Implementation", "Langchain Implementation"],
        index=0,
        help="Choose between the custom RAG implementation or Langchain-based implementation"
    )
    
    # Convert radio selection to boolean for code compatibility
    use_langchain = (rag_framework == "Langchain Implementation")
    
    # Add LLM provider selection
    st.sidebar.markdown("---")
    llm_provider = st.sidebar.radio(
        "Select LLM Provider:",
        ["Mistral", "Gemini"],
        index=0,
        help="Choose which Large Language Model to use for generating responses"
    )
    
    # Convert provider selection to lowercase for code compatibility
    llm_provider = llm_provider.lower()
    
    # Add cross-encoder model selection as radio button (moved from advanced settings)
    st.sidebar.markdown("---")
    cross_encoder_model = st.sidebar.radio(
        "Select Cross-Encoder Reranker:",
        ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-electra-base", "cross-encoder/stsb-roberta-base"],
        index=0,
        format_func=lambda x: {
            "cross-encoder/ms-marco-MiniLM-L-6-v2": "MS-Marco MiniLM (Default, faster)",
            "cross-encoder/ms-marco-electra-base": "MS-Marco Electra (Better accuracy)",
            "cross-encoder/stsb-roberta-base": "STS-B RoBERTa (Best for semantic similarity)"
        }[x],
        help="Select the cross-encoder model for reranking search results"
    )
    
    # Advanced settings expander in sidebar
    with st.sidebar.expander("Advanced Settings"):
        lambda_param = st.slider(
            "Hybrid search balance (λ):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Balance between semantic (1.0) and keyword (0.0) search"
        )
        
        # Define standard section names based on your extract_sections function
        standard_sections = [
            "Financial Statements",
            "Management's Discussion and Analysis",
            "Significant Accounting Policies",
            "Risk Factors",
            "Assets and Liabilities",
            "Cash Flow",
            "Risk Management",
            "Operating Segments",
            "Revenues",
            "Income Taxes",
            "Commitments and Contingencies",
            "Common Stock and Compensation",
            "Shareholders' Equity",
            "Revenue Recognition Standards",
            "Executive Compensation"
        ]
        
        # Add an "All Sections" option at the top
        section_options = ["All Sections"] + standard_sections
        
        section_filter = st.selectbox(
            "Filter by section:",
            options=section_options,
            index=0,
            help="Filter results to specific sections of the 10-K report"
        )
        
        # Set section_filter to None if "All Sections" is selected
        if section_filter == "All Sections":
            section_filter = None
        
        # Show model selection based on the selected LLM provider
        if llm_provider == "mistral":
            model_name = st.selectbox(
                "Mistral Model:",
                ["mistral-medium", "mistral-small", "mistral-large", "mistral-tiny"],
                index=0,
                help="Select Mistral model to use (medium is recommended balance)"
            )
        else:  # Gemini
            model_name = st.selectbox(
                "Gemini Model:",
                ["gemini-2.0-flash", "gemini-1.5-flash"],
                index=0,
                help="Select Gemini model to use (gemini-pro is recommended for text tasks)"
            )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    
    # Update model information to include LLM provider and cross-encoder
    if use_langchain:
        st.sidebar.markdown(f"""
        - **LLM**: {llm_provider.capitalize()} API ({model_name})
        - **Embedding Model**: all-mpnet-base-v2 (via Langchain)
        - **Re-ranker**: {cross_encoder_model.replace("cross-encoder/", "")}
        - **Framework**: Langchain for retrieval
        """)
    else:
        st.sidebar.markdown(f"""
        - **LLM**: {llm_provider.capitalize()} API ({model_name})
        - **Embedding Model**: all-mpnet-base-v2
        - **Re-ranker**: {cross_encoder_model.replace("cross-encoder/", "")}
        """)
    
    # Check and display GPU information
    check_gpu()
    
    # Main content
    st.markdown(f"<h2 style='text-align: center;'>Query {company_name}'s 10-K Report</h2>", unsafe_allow_html=True)
    
    # Example queries
    st.markdown("### Example Queries")
    example_queries = [
        "What are the main revenue sources?",
        "Describe the strategy for international markets",
        "What are the key financial risks?",
        "How does the company approach sustainability?",
        "What are the major competitors mentioned?"
    ]
    
    # Initialize session state for the query if not exists
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Create columns for example query buttons
    cols = st.columns(3)
    
    # Display example query buttons
    for i, example_query in enumerate(example_queries):
        col_idx = i % 3
        if cols[col_idx].button(example_query, key=f"example_{i}"):
            # Only update the text in the query box when example is clicked
            st.session_state.query = example_query
    
    # Query input with session state value
    query = st.text_area("Enter your query:", value=st.session_state.query, height=100, 
                         placeholder="Type your question about the company's 10-K report here...")
    
    # Update session state when user changes the query manually
    st.session_state.query = query
    
    # Process button
    submit_col1, submit_col2 = st.columns([1, 5])
    with submit_col1:
        submit_button = st.button("Submit Query", type="primary")
    
    # Process query only when submit button is clicked
    if submit_button and query:
        start_time = time.time()
        
        # Call the appropriate RAG pipeline with the selected cross-encoder model
        if use_langchain:
            result = langchain_rag_pipeline(
                query, 
                company_name=company_name,
                lambda_param=lambda_param,
                section_filter=section_filter,
                llm_provider=llm_provider,
                model_name=model_name,
                cross_encoder_model=cross_encoder_model
            )
        else:
            result = rag_pipeline(
                query, 
                company_name=company_name,
                lambda_param=lambda_param,
                section_filter=section_filter,
                llm_provider=llm_provider,
                model_name=model_name,
                cross_encoder_model=cross_encoder_model
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Display processing time and models used
        st.info(f"Query processed in {execution_time:.2f} seconds using {llm_provider.capitalize()} ({model_name}) and {cross_encoder_model.replace('cross-encoder/', '')} for reranking")
        
        st.markdown("## Response")
        st.markdown(f'<div class="result-container">{result["response"]}</div>', unsafe_allow_html=True)
        
        # Show retrieved contexts in an expander
        with st.expander("View retrieved contexts"):
            for i, context in enumerate(result['top_2_contexts']):
                st.markdown(f"### Context {i+1}")
                # Try to split on section title if present
                parts = context.split(':\n', 1)  # Split on first occurrence only
                if len(parts) > 1:
                    st.markdown(f"**{parts[0]}**")
                    st.markdown(parts[1])
                else:
                    st.markdown(context)  # Just display the whole context if no split found
                st.markdown("---")
        
        # Advanced results analysis
        with st.expander("View all retrieved documents with context"):
            for i, result_item in enumerate(result.get('retrieved_results', []), 1):
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown(f"**Score: {result_item.get('score', 0):.3f}**")
                    if 'metadata' in result_item:
                        metadata = result_item['metadata']
                        if 'section_title' in metadata:
                            st.markdown(f"**Section: {metadata['section_title']}**")
                        if 'page' in metadata:
                            st.markdown(f"**Page: {metadata['page']}**")
                
                with col2:
                    # Display previous context if available
                    if result_item.get('prev_text'):
                        st.markdown("**Previous Context:**")
                        st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['prev_text']}</div>", unsafe_allow_html=True)
                    
                    # Display main content
                    st.markdown("**Content:**")
                    st.markdown(result_item.get('text', ''))
                    
                    # Display following context if available
                    if result_item.get('next_text'):
                        st.markdown("**Following Context:**")
                        st.markdown(f"<div style='color: #6c757d; font-size: 0.9em;'>{result_item['next_text']}</div>", unsafe_allow_html=True)
                
                st.markdown("---")

if __name__ == "__main__":
    # Set page title and layout
    st.set_page_config(page_title="10-K Report Analysis", layout="wide")
    main()