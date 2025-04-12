# import os
# import re
# import pickle
# import json
# import time
# import requests
# import numpy as np
# import pandas as pd
# import torch
# import faiss
# import streamlit as st
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from PyPDF2 import PdfReader
# import tabula
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# from io import BytesIO

# # Download nltk resources (run once)
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # Set page title and layout
# st.set_page_config(page_title="10-K Report Analysis RAG System", layout="wide")

# # Store API key with an input field
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ""

# # Custom CSS for better styling
# st.markdown("""
# <style>
# .stApp {
#     max-width: 1200px;
#     margin: 0 auto;
# }
# .result-container {
#     background-color: #f8f9fa;
#     padding: 20px;
#     border-radius: 5px;
#     margin-bottom: 20px;
#     color: #333333;  /* Adding explicit dark text color */
# }
# </style>
# """, unsafe_allow_html=True)

# ##################
# # PDF PROCESSING #
# ##################

# def parse_10k_pdf(pdf_file):
#     """
#     Extract text with better handling of layout from a PDF file object
#     """
#     pdf = PdfReader(pdf_file)
#     text = ""

#     # Process page by page to preserve structure better
#     for page in pdf.pages:
#         page_text = page.extract_text()
#         if page_text:
#             # Clean up common PDF extraction issues
#             page_text = re.sub(r'\s+', ' ', page_text)  # Replace multiple spaces
#             page_text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', page_text)  # Handle hyphenation
#             text += page_text + "\n\n"

#     # Extract tables
#     tables = extract_tables_from_pdf(pdf_file)

#     return text, tables

# def extract_tables_from_pdf(pdf_file):
#     """
#     Extract tables from a PDF file object
#     """
#     tables = []
#     try:
#         # Save to a temporary file first (tabula needs a file path)
#         with BytesIO() as temp_file:
#             temp_file.write(pdf_file.getvalue())
#             temp_file.seek(0)
            
#             # More aggressive table extraction settings
#             extracted_tables = tabula.read_pdf(
#                 temp_file,
#                 pages='all',
#                 multiple_tables=True,
#                 lattice=True,  # Try lattice mode for bordered tables
#                 stream=True    # Also try stream mode for unbordered tables
#             )

#             # Process and clean up tables
#             for i, df in enumerate(extracted_tables):
#                 if not df.empty and len(df.columns) > 1:
#                     # Clean up column names
#                     df.columns = [str(col).strip() for col in df.columns]

#                     # Remove completely empty rows and columns
#                     df = df.dropna(how='all').dropna(axis=1, how='all')

#                     # Try to infer better data types
#                     for col in df.columns:
#                         try:
#                             # Try to convert numeric columns
#                             if df[col].dtype == 'object':
#                                 df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='ignore')
#                         except:
#                             pass

#                     if not df.empty:
#                         tables.append(df)
#     except Exception as e:
#         st.error(f"Error extracting tables from PDF: {e}")

#     return tables

# def semantic_chunk_text(text, max_chunk_size=1000, overlap=200):
#     """
#     Split text into semantically meaningful chunks with overlap
#     """
#     # First split into paragraphs
#     paragraphs = re.split(r'\n\s*\n', text)
#     paragraphs = [p.strip() for p in paragraphs if p.strip()]

#     # Extract section titles using regex patterns
#     section_titles = []
#     for p in paragraphs:
#         if re.match(r'^[A-Z\s]{5,}$', p) or re.match(r'^ITEM\s+\d+', p, re.IGNORECASE):
#             section_titles.append(p)
    
#     # Then process paragraphs into semantically meaningful chunks
#     chunks = []
#     current_chunk = ""
#     current_section = "General"  # Default section

#     for paragraph in paragraphs:
#         # Check if this paragraph looks like a section title
#         if any(title in paragraph for title in section_titles):
#             current_section = paragraph
        
#         # If paragraph is very long, split it into sentences
#         if len(paragraph) > max_chunk_size:
#             sentences = sent_tokenize(paragraph)
#             for sentence in sentences:
#                 if len(current_chunk) + len(sentence) < max_chunk_size:
#                     current_chunk += sentence + " "
#                 else:
#                     if current_chunk:
#                         chunks.append((current_chunk.strip(), current_section))
#                     current_chunk = sentence + " "
#         # If adding paragraph doesn't exceed limit, add it
#         elif len(current_chunk) + len(paragraph) < max_chunk_size:
#             current_chunk += paragraph + "\n\n"
#         # Otherwise start a new chunk
#         else:
#             if current_chunk:
#                 chunks.append((current_chunk.strip(), current_section))
#             current_chunk = paragraph + "\n\n"

#     # Add the last chunk if it has content
#     if current_chunk:
#         chunks.append((current_chunk.strip(), current_section))

#     # Create overlapping chunks for better context preservation
#     overlapping_chunks = []
#     for i in range(len(chunks)):
#         chunk, section = chunks[i]

#         # Add overlap with previous chunk if possible
#         prev_overlap = ""
#         if i > 0:
#             prev_chunk, _ = chunks[i-1]
#             # Get last N characters from previous chunk
#             prev_overlap = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk

#         # Add overlap with next chunk if possible
#         next_overlap = ""
#         if i < len(chunks) - 1:
#             next_chunk, _ = chunks[i+1]
#             # Get first N characters from next chunk
#             next_overlap = next_chunk[:overlap] if len(next_chunk) > overlap else next_chunk

#         overlapping_chunks.append({
#             'text': chunk,
#             'section_title': section,
#             'prev_overlap': prev_overlap,
#             'next_overlap': next_overlap,
#             'chunk_id': i,
#             'type': 'text'
#         })

#     return overlapping_chunks

# def tables_to_text(tables, base_chunk_id=0):
#     """
#     Convert tables to enhanced text format for searching
#     """
#     table_chunks = []
#     for i, table in enumerate(tables):
#         try:
#             # Convert DataFrame to more readable text format
#             # Include a header with column names
#             header = " | ".join(table.columns)
#             rows = []

#             # Format each row
#             for _, row in table.iterrows():
#                 formatted_row = " | ".join([str(val) for val in row.values])
#                 rows.append(formatted_row)

#             # Combine with descriptive text
#             table_text = f"Table {i+1}:\nColumns: {header}\n" + "\n".join(rows)
            
#             table_chunks.append({
#                 'text': table_text,
#                 'section_title': "Financial Tables",
#                 'chunk_id': base_chunk_id + i,
#                 'type': 'table',
#                 'prev_overlap': "",
#                 'next_overlap': ""
#             })

#             # Also create row-by-row descriptions for complex tables
#             if len(table) > 1 and len(table.columns) > 2:
#                 for j, (_, row) in enumerate(table.iterrows()):
#                     row_desc = f"Table {i+1}, Row {j+1}: "
#                     for col in table.columns:
#                         row_desc += f"{col}: {row[col]}, "
#                     row_desc = row_desc.rstrip(", ")
                    
#                     table_chunks.append({
#                         'text': row_desc,
#                         'section_title': "Financial Tables",
#                         'chunk_id': base_chunk_id + i + j + 1000,  # Offset to avoid collisions
#                         'type': 'table_row',
#                         'prev_overlap': "",
#                         'next_overlap': ""
#                     })
#         except Exception as e:
#             st.warning(f"Error converting table {i+1} to text: {e}")

#     return table_chunks

# def extract_sections(chunks):
#     """
#     Categorize chunks into standard 10-K report sections
#     """
#     # Define keywords that indicate specific sections
#     section_keywords = {
#         "Financial Statements": ["statement of operations", "balance sheet", "cash flow", "financial statement", "consolidated statements"],
#         "Management's Discussion and Analysis": ["management's discussion", "MD&A", "results of operations"],
#         "Risk Factors": ["risk factor", "risks", "uncertainties"],
#         "Executive Compensation": ["executive compensation", "officer compensation", "director compensation"],
#         "Assets and Liabilities": ["assets", "liabilities", "balance sheet"],
#         "Cash Flow": ["cash flow", "cash provided by"],
#         "Revenue Recognition": ["revenue recognition"],
#         "Operating Segments": ["segment", "business segment", "reportable segment"],
#         "Significant Accounting Policies": ["accounting policies", "significant accounting", "accounting standards"],
#     }

#     # Update each chunk with a more specific section categorization
#     for chunk in chunks:
#         chunk_text = chunk['text'].lower()
        
#         # Start with the manually assigned section_title
#         current_section = chunk.get('section_title', "General")
        
#         # Try to find a more specific standard section based on keywords
#         for section_name, keywords in section_keywords.items():
#             if any(keyword in chunk_text for keyword in keywords):
#                 current_section = section_name
#                 break
                
#         chunk['section_title'] = current_section
    
#     return chunks

# def create_search_index(chunks, company_name):
#     """
#     Create FAISS index and other search components
#     """
#     # Get just the text content for embedding
#     texts = [chunk['text'] for chunk in chunks]
    
#     # 1. Create embeddings with a more powerful model
#     with st.spinner("Generating embeddings with all-mpnet-base-v2..."):
#         model = SentenceTransformer('all-mpnet-base-v2')
#         embeddings = model.encode(texts, show_progress_bar=True)

#     # Convert to float32 (required by FAISS)
#     embeddings = np.array(embeddings).astype('float32')
#     st.success(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

#     # 2. Create FAISS index with better similarity search
#     dimension = embeddings.shape[1]
    
#     with st.spinner("Building FAISS index..."):
#         # Use IVF index for larger collections (faster search)
#         if len(chunks) > 1000:
#             # Number of clusters - rule of thumb: sqrt(n) where n is the number of vectors
#             nlist = int(np.sqrt(len(chunks)))
#             quantizer = faiss.IndexFlatL2(dimension)
#             index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
#             # Need to train IVF index
#             index.train(embeddings)
#             index.add(embeddings)
#         else:
#             # For smaller collections, use flat index
#             index = faiss.IndexFlatL2(dimension)
#             index.add(embeddings)

#     # 3. Create sparse BM25 index for keyword-based retrieval
#     with st.spinner("Building BM25 index for keyword search..."):
#         tokenized_chunks = [chunk['text'].split() for chunk in chunks]
#         bm25 = BM25Okapi(tokenized_chunks)

#     return index, bm25, model, chunks

# def save_search_index(index, chunks, bm25, output_dir, company_name):
#     """
#     Save search index components to disk
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save FAISS index
#     faiss.write_index(index, os.path.join(output_dir, f"{company_name.lower()}_10k_enhanced_index.faiss"))
    
#     # Save metadata (chunks)
#     with open(os.path.join(output_dir, f"{company_name.lower()}_10k_enhanced_metadata.pkl"), "wb") as f:
#         pickle.dump(chunks, f)
    
#     # Save BM25 model
#     with open(os.path.join(output_dir, f"{company_name.lower()}_10k_bm25.pkl"), "wb") as f:
#         pickle.dump(bm25, f)
    
#     st.success(f"Saved search index to '{output_dir}' directory")

# ##################
# # SEARCH & RAG   #
# ##################

# @st.cache_resource
# def load_rag_components(company_name):
#     """
#     Load the pre-processed RAG components with caching for better performance
#     Uses folder structure: Company 10-K's/{company_name}/{files}
#     """
#     base_path = f"Company 10-K's/{company_name}"
    
#     with st.spinner(f"Loading RAG components for {company_name}..."):
#         # Load FAISS index
#         index = faiss.read_index(f"{base_path}/{company_name.lower()}_10k_enhanced_index.faiss")
        
#         # Load metadata
#         with open(f"{base_path}/{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
#             metadata = pickle.load(f)
        
#         # Load BM25 model
#         with open(f"{base_path}/{company_name.lower()}_10k_bm25.pkl", "rb") as f:
#             bm25_model = pickle.load(f)
        
#         # Load embedding model
#         embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
#         # Load Cross Encoder
#         cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
#         return index, metadata, embedding_model, bm25_model, cross_encoder

# def mmr(sorted_results, metadata, embedding_model, top_k, diversity=0.3):
#     """
#     Apply Maximum Marginal Relevance to reduce redundancy in results
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
#     """
#     # Apply section filter if provided
#     if section_filter:
#         filtered_indices = [i for i, chunk in enumerate(metadata)
#                            if chunk.get('section_title') == section_filter or
#                               chunk.get('type') == section_filter]

#         if not filtered_indices:
#             st.warning(f"No chunks found with section filter: {section_filter}")
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
#             # Use stored overlap instead of looking up previous and next chunks
#             prev_text = chunk.get('prev_overlap', '')
#             next_text = chunk.get('next_overlap', '')

#             context = {
#                 'text': chunk['text'],
#                 'metadata': chunk,
#                 'score': score,
#                 'prev_text': prev_text,
#                 'next_text': next_text
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

# def generate_mistral_response(query, retrieved_context, api_key, company_name):
#     """
#     Generate a response using Mistral API with retrieved contexts
#     """
#     # Construct the prompt with contexts but instruct not to mention context numbers
#     prompt = f"""You are an expert assistant analyzing {company_name}'s 10-K report.
# Use the following contexts to answer the query precisely and comprehensively.

# Query: {query}

# Retrieved Contexts:
# {retrieved_context}

# Important Guidelines:
# 1. Base your response STRICTLY on the provided contexts
# 2. Do not introduce information not present in these contexts
# 3. If the context includes previous or next context indicators, incorporate that information appropriately
# 4. Provide a detailed and accurate response
# 5. DO NOT explicitly mention "Context 1" or "Context 2" in your response - just provide a coherent answer without referencing the source numbering
# 6. Present the information as a unified, seamless response
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

# def rag_pipeline(query, base_path, company_name, lambda_param=0.5, section_filter=None, api_key=None):
#     """
#     Full RAG pipeline: retrieve, generate, and return response
#     """
#     try:
#         # Load RAG components
#         with st.spinner("Loading search index..."):
#             # Load FAISS index
#             index = faiss.read_index(f"{base_path}/{company_name.lower()}_10k_enhanced_index.faiss")
            
#             # Load metadata
#             with open(f"{base_path}/{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
#                 metadata = pickle.load(f)
            
#             # Load BM25 model
#             with open(f"{base_path}/{company_name.lower()}_10k_bm25.pkl", "rb") as f:
#                 bm25_model = pickle.load(f)
            
#             # Load embedding model
#             embedding_model = SentenceTransformer('all-mpnet-base-v2')
            
#             # Load Cross Encoder
#             cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
#         # Perform hybrid search
#         with st.spinner("Retrieving relevant information..."):
#             retrieved_results = hybrid_search(
#                 query, index, metadata, embedding_model, bm25_model,
#                 cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
#                 section_filter=section_filter
#             )
        
#         # Format top 2 contexts
#         context = format_retrieved_context(retrieved_results)
        
#         # Generate response using Mistral API
#         with st.spinner("Generating response with Mistral API..."):
#             if api_key:
#                 response = generate_mistral_response(query, context, api_key, company_name)
#             else:
#                 response = "Please enter a valid Mistral API key to generate responses."
        
#         return {
#             'query': query,
#             'retrieved_results': retrieved_results,
#             'top_2_contexts': context.split('\n\n'),
#             'response': response
#         }
#     except Exception as e:
#         st.error(f"Error in RAG pipeline: {str(e)}")
#         return {
#             'query': query,
#             'retrieved_results': [],
#             'top_2_contexts': [],
#             'response': f"Error: {str(e)}"
#         }

# def check_gpu():
#     if torch.cuda.is_available():
#         gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
#         memory_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
#         st.sidebar.success(f"✅ Using GPU for embeddings\n{gpu_info}\n{memory_info}")
#     else:
#         st.sidebar.warning("⚠️ GPU not detected. Using CPU for embeddings (may be slower)")

# ##################
# # UI COMPONENTS  #
# ##################

# def upload_section():
#     st.title("10-K Report Analysis System")
#     st.markdown("""
#     This application allows you to extract information from 10-K reports and create a custom search index
#     for retrieval-augmented generation (RAG). Upload your 10-K PDF file to get started.
#     """)
    
#     # File uploader
#     uploaded_file = st.file_uploader("Upload 10-K PDF", type="pdf")
    
#     if uploaded_file is not None:
#         # Company name input
#         company_name = st.text_input("Enter company name (e.g., NIKE, APPLE):", 
#                                      value=uploaded_file.name.split('.')[0].upper())
        
#         # Process button
#         process_button = st.button("Process 10-K Report")
        
#         if process_button:
#             if not company_name:
#                 st.error("Please enter a company name before processing.")
#                 return
                
#             with st.spinner("Processing PDF file..."):
#                 # Step 1: Extract text and tables
#                 text, tables = parse_10k_pdf(uploaded_file)
                
#                 # Show statistics
#                 st.write(f"Extracted {len(text)} characters of text")
#                 st.write(f"Extracted {len(tables)} tables")
                
#                 # Preview first 500 characters
#                 st.subheader("Text Preview")
#                 st.markdown(f"```\n{text[:500]}...\n```")
                
#                 if tables:
#                     st.subheader("Table Preview")
#                     st.dataframe(tables[0].head())
                
#                 # Step 2: Create chunks
#                 text_chunks = semantic_chunk_text(text)
#                 st.write(f"Created {len(text_chunks)} text chunks")
                
#                 # Step 3: Convert tables to text
#                 table_chunks = tables_to_text(tables, base_chunk_id=len(text_chunks))
#                 st.write(f"Created {len(table_chunks)} table chunks")
                
#                 # Step 4: Combine and identify sections
#                 all_chunks = text_chunks + table_chunks
#                 all_chunks = extract_sections(all_chunks)
#                 st.write(f"Total chunks: {len(all_chunks)}")
                
#                 # Step 5: Create search index
#                 index, bm25, model, chunks = create_search_index(all_chunks, company_name)
                
#                 # Step 6: Save everything to disk
#                 output_dir = f"Company 10-K's/{company_name}"
#                 save_search_index(index, chunks, bm25, output_dir, company_name)
                
#                 # Show success message with next steps
#                 st.success(f"""
#                 Successfully processed {company_name}'s 10-K report!
                
#                 Your search index has been created and saved to:
#                 - {output_dir}/{company_name.lower()}_10k_enhanced_index.faiss
#                 - {output_dir}/{company_name.lower()}_10k_enhanced_metadata.pkl
#                 - {output_dir}/{company_name.lower()}_10k_bm25.pkl
                
#                 You can now switch to the "Query" tab to ask questions about the 10-K report.
#                 """)
                
#                 # Store this company name in session state for easy access in query tab
#                 st.session_state.last_processed_company = company_name
                
#                 # Add a button to switch to query tab
#                 if st.button("Go to Query Tab"):
#                     st.session_state.active_tab = "Query"
#                     st.experimental_rerun()

# def query_section():
#     st.title("Query 10-K Report")
    
#     # API Key input in sidebar
#     st.sidebar.title("Mistral API Settings")
#     api_key = st.sidebar.text_input("Enter Mistral API Key:", value=st.session_state.api_key, type="password")
#     st.session_state.api_key = api_key  # Store API key in session state
    
#     # Company selection
#     # List all directories in the "Company 10-K's" folder
#     try:
#         company_dirs = [d for d in os.listdir("Company 10-K's") if os.path.isdir(os.path.join("Company 10-K's", d))]
#         if not company_dirs:
#             st.warning("No processed 10-K reports found. Please upload and process a 10-K report first.")
#             return
#     except FileNotFoundError:
#         st.warning("No processed 10-K reports found. Please upload and process a 10-K report first.")
#         return
    
#     # Default to last processed company if available
#     default_company = st.session_state.get('last_processed_company', company_dirs[0]) if company_dirs else ""
    
#     # Company selection dropdown
#     selected_company = st.selectbox("Select company:", company_dirs, 
#                                    index=company_dirs.index(default_company) if default_company in company_dirs else 0)
    
#     # Base path for the selected company
#     base_path = f"Company 10-K's/{selected_company}"
    
#     # Section filter options - load available sections from metadata
#     try:
#         with open(f"{base_path}/{selected_company.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
#             metadata = pickle.load(f)
            
#         # Extract unique section titles
#         section_titles = sorted(list(set(chunk.get('section_title', "General") for chunk in metadata)))
        
#         # Also add table types as filter options
#         section_titles.extend(["table", "table_row"])
        
#         # Remove duplicates
#         section_titles = sorted(list(set(section_titles)))
#     except:
#         section_titles = ["All Sections"]
    
#     # Advanced options expander
#     with st.expander("Advanced Options"):
#         # Section filter
#         section_filter = st.selectbox("Filter by section:", ["All Sections"] + section_titles)
#         if section_filter == "All Sections":
#             section_filter = None
            
#         # Lambda parameter (balance between semantic and keyword search)
#         lambda_param = st.slider(
#             "Semantic vs. Keyword Balance:", 
#             min_value=0.0, 
#             max_value=1.0, 
#             value=0.6,
#             help="Higher values favor semantic search, lower values favor keyword search"
#         )
    
#     # Query input
#     query = st.text_area("Enter your question about the 10-K report:", 
#                         placeholder="Example: What are the major risk factors mentioned in the report?")
    
#     # Submit button
#     submit_button = st.button("Submit Query")
    
#     # Process query when submit button is clicked
#     if submit_button and query:
#         # Check for API key
#         if not api_key:
#             st.warning("Please enter your Mistral API key in the sidebar.")
#             return
            
#         try:
#             # Run RAG pipeline
#             results = rag_pipeline(
#                 query=query,
#                 base_path=base_path,
#                 company_name=selected_company,
#                 lambda_param=lambda_param,
#                 section_filter=section_filter,
#                 api_key=api_key
#             )
            
#             # Display results
#             st.markdown("### Response")
#             st.markdown(f"""<div class="result-container">{results['response']}</div>""", unsafe_allow_html=True)
            
#             # Show source contexts in an expander
#             with st.expander("Show Source Contexts"):
#                 st.markdown("### Retrieved Contexts")
                
#                 # Display each retrieved context
#                 for i, result in enumerate(results['retrieved_results'], 1):
#                     st.markdown(f"**Context {i}** (Score: {result['score']:.2f})")
#                     st.markdown(f"**Section:** {result['metadata'].get('section_title', 'General')}")
#                     st.markdown(f"**Type:** {result['metadata'].get('type', 'text')}")
                    
#                     # Display the text with markdown formatting
#                     text_to_display = result['text'].replace('\n', '\n\n')
#                     st.markdown(f"""```{text_to_display}```""")
#                     st.divider()
                    
#         except Exception as e:
#             st.error(f"Error processing query: {str(e)}")

# def main():
#     # Check for GPU
#     check_gpu()
    
#     # Create tab layout
#     st.sidebar.title("Navigation")
#     tabs = ["Upload", "Query"]
    
#     # Use session state to remember active tab
#     if 'active_tab' not in st.session_state:
#         st.session_state.active_tab = "Upload"
    
#     # Tab selection using radio buttons in sidebar
#     selected_tab = st.sidebar.radio("Select Action:", tabs, 
#                                    index=tabs.index(st.session_state.active_tab))
    
#     # Update session state
#     st.session_state.active_tab = selected_tab
    
#     # Display appropriate section based on selected tab
#     if selected_tab == "Upload":
#         upload_section()
#     else:
#         query_section()
    
#     # Add additional information in the sidebar
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("""
#     ### About This App
#     This application uses advanced RAG techniques to analyze 10-K reports:
#     * PDF parsing with text and table extraction
#     * Semantic chunking of documents
#     * Hybrid search (BM25 + FAISS)
#     * Maximum Marginal Relevance for diversity
#     * Cross-encoder reranking
#     * Mistral AI for response generation
#     """)
    
#     # Link to instructions or help
#     st.sidebar.markdown("---")
#     st.sidebar.info("""
#     **Instructions:**
#     1. Upload your 10-K PDF in the "Upload" tab
#     2. Enter company name and process the report
#     3. Switch to "Query" tab to ask questions
#     4. Enter your Mistral AI API key in the sidebar
#     """)

# if __name__ == "__main__":
#     main()



import os
import re
import pickle
import json
import time
import requests
import numpy as np
import pandas as pd
import torch
import faiss
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from PyPDF2 import PdfReader
import tabula
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from io import BytesIO
import random
import threading
import yfinance as yf
import pandas as pd
from fuzzywuzzy import process
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import tempfile
import os
import os
import re
import json
import pickle
import time
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Download nltk resources (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')



# Set your Mistral API key here (not visible to users)
MISTRAL_API_KEY = "grL3IacMpVhmjmfNW6ZNBismxLyNS7gV"  # Replace with your actual API key
GEMINI_API_KEY = "AIzaSyAfxHwFSHC-5_h3BQ8GR0N1H6FAfyJqqUI"  # Add this line

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)


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

##################
# PDF PROCESSING #
##################

def parse_10k_pdf(pdf_file):
    """
    Extract text with better handling of layout from a PDF file object
    """
    pdf = PdfReader(pdf_file)
    text = ""

    # Process page by page to preserve structure better
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            # Clean up common PDF extraction issues
            page_text = re.sub(r'\s+', ' ', page_text)  # Replace multiple spaces
            page_text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', page_text)  # Handle hyphenation
            text += page_text + "\n\n"

    # Extract tables
    tables = extract_tables_from_pdf(pdf_file)

    return text, tables

def extract_tables_from_pdf(pdf_file):
    """
    Extract tables from a PDF file object
    """
    tables = []
    
    # Save to a temporary file first (tabula needs a file path)
    with BytesIO() as temp_file:
        temp_file.write(pdf_file.getvalue())
        temp_file.seek(0)
                  
        # More aggressive table extraction settings
        extracted_tables = tabula.read_pdf(
            temp_file,
            pages='all',
            multiple_tables=True,
            lattice=True,  # Try lattice mode for bordered tables
            stream=True    # Also try stream mode for unbordered tables
        )
        
        # Process and clean up tables
        for i, df in enumerate(extracted_tables):
            if not df.empty and len(df.columns) > 1:
                # Clean up column names
                df.columns = [str(col).strip() for col in df.columns]
                
                # Remove completely empty rows and columns
                df = df.dropna(how='all').dropna(axis=1, how='all')
                
                # Try to infer better data types
                for col in df.columns:
                    try:
                        # Try to convert numeric columns
                        if df[col].dtype == 'object':
                            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('$', '').str.replace('%', ''), errors='ignore')
                    except:
                        pass
                
                if not df.empty:
                    tables.append(df)
    
    return tables

def semantic_chunk_text(text, max_chunk_size=1000, overlap=200):
    """
    Split text into semantically meaningful chunks with overlap
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of chunk dictionaries
    """
    # First split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Extract section titles using regex patterns
    section_titles = []
    for p in paragraphs:
        if re.match(r'^[A-Z\s]{5,}$', p) or re.match(r'^ITEM\s+\d+', p, re.IGNORECASE):
            section_titles.append(p)
    
    # Then process paragraphs into semantically meaningful chunks
    chunks = []
    current_chunk = ""
    current_section = "General"  # Default section
    
    for paragraph in paragraphs:
        # Check if this paragraph looks like a section title
        if any(title in paragraph for title in section_titles):
            current_section = paragraph
            
        # If paragraph is very long, split it into sentences
        if len(paragraph) > max_chunk_size:
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append((current_chunk.strip(), current_section))
                    current_chunk = sentence + " "
        # If adding paragraph doesn't exceed limit, add it
        elif len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        # Otherwise start a new chunk
        else:
            if current_chunk:
                chunks.append((current_chunk.strip(), current_section))
            current_chunk = paragraph + "\n\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append((current_chunk.strip(), current_section))
    
    # Create overlapping chunks for better context preservation
    overlapping_chunks = []
    for i in range(len(chunks)):
        chunk, section = chunks[i]
        
        # Add overlap with previous chunk if possible
        prev_overlap = ""
        if i > 0:
            prev_chunk, _ = chunks[i-1]
            # Get last N characters from previous chunk
            prev_overlap = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
        
        # Add overlap with next chunk if possible
        next_overlap = ""
        if i < len(chunks) - 1:
            next_chunk, _ = chunks[i+1]
            # Get first N characters from next chunk
            next_overlap = next_chunk[:overlap] if len(next_chunk) > overlap else next_chunk
        
        overlapping_chunks.append({
            'text': chunk,
            'section_title': section,
            'prev_overlap': prev_overlap,
            'next_overlap': next_overlap,
            'chunk_id': i,
            'type': 'text'
        })
    
    return overlapping_chunks

def tables_to_text(tables, base_chunk_id=0):
    """
    Convert tables to enhanced text format for searching
    """
    table_chunks = []
    for i, table in enumerate(tables):
        try:
            # Convert DataFrame to more readable text format
            # Include a header with column names
            header = " | ".join(table.columns)
            rows = []

            # Format each row
            for _, row in table.iterrows():
                formatted_row = " | ".join([str(val) for val in row.values])
                rows.append(formatted_row)

            # Combine with descriptive text
            table_text = f"Table {i+1}:\nColumns: {header}\n" + "\n".join(rows)
            
            table_chunks.append({
                'text': table_text,
                'section_title': "Financial Tables",
                'chunk_id': base_chunk_id + i,
                'type': 'table',
                'prev_overlap': "",
                'next_overlap': ""
            })

            # Also create row-by-row descriptions for complex tables
            if len(table) > 1 and len(table.columns) > 2:
                for j, (_, row) in enumerate(table.iterrows()):
                    row_desc = f"Table {i+1}, Row {j+1}: "
                    for col in table.columns:
                        row_desc += f"{col}: {row[col]}, "
                    row_desc = row_desc.rstrip(", ")
                    
                    table_chunks.append({
                        'text': row_desc,
                        'section_title': "Financial Tables",
                        'chunk_id': base_chunk_id + i + j + 1000,  # Offset to avoid collisions
                        'type': 'table_row',
                        'prev_overlap': "",
                        'next_overlap': ""
                    })
        except Exception as e:
            st.warning(f"Error converting table {i+1} to text: {e}")

    return table_chunks

def extract_sections(chunks):
    """
    Categorize chunks into standard 10-K report sections
    """
    # Define keywords that indicate specific sections
    section_keywords = {
        "Financial Statements": ["statement of operations", "balance sheet", "cash flow", "financial statement", "consolidated statements"],
        "Management's Discussion and Analysis": ["management's discussion", "MD&A", "results of operations"],
        "Risk Factors": ["risk factor", "risks", "uncertainties"],
        "Executive Compensation": ["executive compensation", "officer compensation", "director compensation"],
        "Assets and Liabilities": ["assets", "liabilities", "balance sheet"],
        "Cash Flow": ["cash flow", "cash provided by"],
        "Revenue Recognition": ["revenue recognition"],
        "Operating Segments": ["segment", "business segment", "reportable segment"],
        "Significant Accounting Policies": ["accounting policies", "significant accounting", "accounting standards"],
    }

    # Update each chunk with a more specific section categorization
    for chunk in chunks:
        chunk_text = chunk['text'].lower()
        
        # Start with the manually assigned section_title
        current_section = chunk.get('section_title', "General")
        
        # Try to find a more specific standard section based on keywords
        for section_name, keywords in section_keywords.items():
            if any(keyword in chunk_text for keyword in keywords):
                current_section = section_name
                break
                
        chunk['section_title'] = current_section
    
    return chunks

def create_search_index(chunks, company_name, embedding_model_name='all-mpnet-base-v2'):
    """
    Create FAISS index and other search components with configurable embedding model
    
    Args:
        chunks: List of text chunks to index
        company_name: Name of the company for the data
        embedding_model_name: Name of the embedding model to use (default: 'all-mpnet-base-v2')
    """
    # Get just the text content for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    # 1. Create embeddings with the selected model
    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Convert to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    
    # 2. Create FAISS index with better similarity search
    dimension = embeddings.shape[1]
    
    # Use IVF index for larger collections (faster search)
    if len(chunks) > 1000:
        # Number of clusters - rule of thumb: sqrt(n) where n is the number of vectors
        nlist = int(np.sqrt(len(chunks)))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # Need to train IVF index
        index.train(embeddings)
        index.add(embeddings)
    else:
        # For smaller collections, use flat index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    
    # 3. Create sparse BM25 index for keyword-based retrieval
    tokenized_chunks = [chunk['text'].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    return index, bm25, model, chunks

def save_search_index(index, chunks, bm25, output_dir, company_name):
    """
    Save search index components to disk
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, f"{company_name.lower()}_10k_enhanced_index.faiss"))
    
    # Save metadata (chunks)
    with open(os.path.join(output_dir, f"{company_name.lower()}_10k_enhanced_metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    
    # Save BM25 model
    with open(os.path.join(output_dir, f"{company_name.lower()}_10k_bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)


def fixed_length_chunk_text(text, chunk_size=1000):
    """
    Split text into fixed-length chunks without overlap
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of chunk dictionaries
    """
    # First split into paragraphs to preserve some structure
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    current_section = "General"  # Default section
    chunk_id = 0
    
    for paragraph in paragraphs:
        # Check if this paragraph looks like a section title
        if re.match(r'^[A-Z\s]{5,}$', paragraph) or re.match(r'^ITEM\s+\d+', paragraph, re.IGNORECASE):
            current_section = paragraph
            
        # If adding paragraph would exceed limit, start a new chunk
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            # If current chunk has content, add it to chunks
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'section_title': current_section,
                    'chunk_id': chunk_id,
                    'type': 'text',
                    'prev_overlap': "",
                    'next_overlap': ""
                })
                chunk_id += 1
                current_chunk = ""
            
            # If paragraph itself is longer than chunk_size, split it
            if len(paragraph) > chunk_size:
                # Split into parts
                paragraph_parts = [paragraph[i:i+chunk_size] for i in range(0, len(paragraph), chunk_size)]
                for part in paragraph_parts:
                    chunks.append({
                        'text': part.strip(),
                        'section_title': current_section,
                        'chunk_id': chunk_id,
                        'type': 'text',
                        'prev_overlap': "",
                        'next_overlap': ""
                    })
                    chunk_id += 1
            else:
                # Start a new chunk with this paragraph
                current_chunk = paragraph + "\n\n"
        else:
            # Add paragraph to current chunk
            current_chunk += paragraph + "\n\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'section_title': current_section,
            'chunk_id': chunk_id,
            'type': 'text',
            'prev_overlap': "",
            'next_overlap': ""
        })
    
    return chunks

def tfidf_chunk_text(text, max_chunk_size=1000, min_chunk_size=200):
    """
    Split text into chunks based on TF-IDF to ensure semantically related content stays together
    
    Args:
        text: The text to split
        max_chunk_size: Maximum size of each chunk
        min_chunk_size: Minimum size of each chunk
        
    Returns:
        List of chunk dictionaries
    """
    # First split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Identify section titles
    section_titles = []
    for p in paragraphs:
        if re.match(r'^[A-Z\s]{5,}$', p) or re.match(r'^ITEM\s+\d+', p, re.IGNORECASE):
            section_titles.append(p)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    
    # Only proceed with TF-IDF if we have enough paragraphs
    if len(paragraphs) >= 5:  # Need some minimum number for meaningful TF-IDF
        try:
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            
            # Calculate similarity between consecutive paragraphs
            similarity_scores = []
            for i in range(len(paragraphs) - 1):
                score = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[i+1:i+2]
                )[0][0]
                similarity_scores.append(score)
            
            # Find natural break points (low similarity between paragraphs)
            # We'll use these as chunk boundaries
            break_points = []
            for i, score in enumerate(similarity_scores):
                # Consider paragraphs with low similarity as potential break points
                if score < 0.2:  # Threshold for dissimilarity
                    break_points.append(i + 1)  # +1 because we want the index of the next paragraph
        except:
            # Fall back to standard chunking if TF-IDF fails
            break_points = []
    else:
        break_points = []
    
    # Create chunks based on break points and size constraints
    chunks = []
    current_chunk = ""
    current_section = "General"  # Default section
    current_paragraphs = []
    chunk_id = 0
    
    for i, paragraph in enumerate(paragraphs):
        # Check if this paragraph looks like a section title
        if any(title in paragraph for title in section_titles):
            # Section titles create natural break points
            if current_paragraphs:
                # Process the chunk we've accumulated so far
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append({
                    'text': chunk_text.strip(),
                    'section_title': current_section,
                    'chunk_id': chunk_id,
                    'type': 'text',
                    'prev_overlap': "",
                    'next_overlap': ""
                })
                chunk_id += 1
                current_paragraphs = []
            
            current_section = paragraph
            current_paragraphs.append(paragraph)
        else:
            # Check if we're at a break point or would exceed max size
            current_size = sum(len(p) for p in current_paragraphs) + len(paragraph) + 2 * len(current_paragraphs)
            
            if (i in break_points and current_size >= min_chunk_size) or current_size + len(paragraph) > max_chunk_size:
                # Create a chunk from accumulated paragraphs if we have enough content
                if current_paragraphs:
                    chunk_text = "\n\n".join(current_paragraphs)
                    chunks.append({
                        'text': chunk_text.strip(),
                        'section_title': current_section,
                        'chunk_id': chunk_id,
                        'type': 'text',
                        'prev_overlap': "",
                        'next_overlap': ""
                    })
                    chunk_id += 1
                    current_paragraphs = []
            
            # Add the current paragraph to our accumulating list
            current_paragraphs.append(paragraph)
    
    # Add the last chunk if it has content
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append({
            'text': chunk_text.strip(),
            'section_title': current_section,
            'chunk_id': chunk_id,
            'type': 'text',
            'prev_overlap': "",
            'next_overlap': ""
        })
    
    return chunks

##################
# SEARCH & RAG   #
##################

@st.cache_resource
def load_rag_components(company_name, embedding_model_name='all-mpnet-base-v2', cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Load the pre-processed RAG components with caching for better performance
    Uses folder structure: Company 10-K's/{company_name}/{files}
    Args:
        company_name: Name of the company for the data
        embedding_model_name: Name of the embedding model to use (default: 'all-mpnet-base-v2')
        cross_encoder_name: Name of the cross-encoder model to use (default: 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    """
    base_path = f"Company 10-K's/{company_name}"
    
    # Load FAISS index
    index = faiss.read_index(f"{base_path}/{company_name.lower()}_10k_enhanced_index.faiss")
    
    # Load metadata
    with open(f"{base_path}/{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # Load BM25 model
    with open(f"{base_path}/{company_name.lower()}_10k_bm25.pkl", "rb") as f:
        bm25_model = pickle.load(f)
    
    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Load Cross Encoder
    cross_encoder = CrossEncoder(cross_encoder_name)
    
    return index, metadata, embedding_model, bm25_model, cross_encoder

def mmr(sorted_results, metadata, embedding_model, top_k, diversity=0.3):
    """
    Apply Maximum Marginal Relevance to reduce redundancy in results
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
    """
    # Apply section filter if provided
    if section_filter:
        filtered_indices = [i for i, chunk in enumerate(metadata)
                           if chunk.get('section_title') == section_filter or
                              chunk.get('type') == section_filter]

        if not filtered_indices:
            st.warning(f"No chunks found with section filter: {section_filter}")
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
            # Use stored overlap instead of looking up previous and next chunks
            prev_text = chunk.get('prev_overlap', '')
            next_text = chunk.get('next_overlap', '')

            context = {
                'text': chunk['text'],
                'metadata': chunk,
                'score': score,
                'prev_text': prev_text,
                'next_text': next_text
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
    """
    Select the top N most relevant contexts based on score
    """
    # Sort results by score in descending order
    sorted_results = sorted(retrieved_results, key=lambda x: x['score'], reverse=True)
    
    # Select top contexts (no filtering by score value)
    top_contexts = sorted_results[:num_contexts]
    
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

def generate_mistral_response(query, retrieved_context, api_key, company_name):
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
        "model": "mistral-medium",  # You can change to other models: mistral-tiny, mistral-small, mistral-large
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

def generate_gemini_response(query, retrieved_context, company_name):
    """
    Generate a response using Gemini API with retrieved contexts
    """
    # Configure Gemini (assuming this is done at the top of your script)
    # genai.configure(api_key=GEMINI_API_KEY)
    
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

    # Create Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Send request to Gemini API
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"
    
def rag_pipeline(query, base_path, company_name, lambda_param=0.5, section_filter=None, llm_model="mistral", embedding_model_name='all-mpnet-base-v2', cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """
    Full RAG pipeline: retrieve, generate, and return response with model selection

    Args:
        query: User query
        base_path: Base path to the company data
        company_name: Name of the company
        lambda_param: Weight for hybrid search (0.5 means equal weight)
        section_filter: Optional filter to limit search to specific sections
        llm_model: LLM to use for generation ("mistral" or "gemini")
        embedding_model_name: Embedding model to use ('all-mpnet-base-v2' or 'finbert')
        cross_encoder_name: Cross-encoder model to use for reranking
    """
    try:
        # Load RAG components
        # Load FAISS index
        index = faiss.read_index(f"{base_path}/{company_name.lower()}_10k_enhanced_index.faiss")

        # Load metadata
        with open(f"{base_path}/{company_name.lower()}_10k_enhanced_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # Load BM25 model
        with open(f"{base_path}/{company_name.lower()}_10k_bm25.pkl", "rb") as f:
            bm25_model = pickle.load(f)

        # Load embedding model based on user selection
        embedding_model = SentenceTransformer(embedding_model_name)

        # Load Cross Encoder based on user selection
        cross_encoder = CrossEncoder(cross_encoder_name)

        # Perform hybrid search
        retrieved_results = hybrid_search(
            query, index, metadata, embedding_model, bm25_model,
            cross_encoder=cross_encoder, k=5, lambda_param=lambda_param,
            section_filter=section_filter
        )

        # Format top 2 contexts
        context = format_retrieved_context(retrieved_results)

        # Generate response using selected model
        if llm_model.lower() == "gemini":
            response = generate_gemini_response(query, context, company_name)
        else:  # Default to Mistral
            response = generate_mistral_response(query, context, MISTRAL_API_KEY, company_name)

        return {
            'query': query,
            'retrieved_results': retrieved_results,
            'top_2_contexts': context.split('\n\n'),
            'response': response,
            'model_used': llm_model,
            'embedding_model': embedding_model_name,
            'cross_encoder': cross_encoder_name
        }
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return {
            'query': query,
            'retrieved_results': [],
            'top_2_contexts': [],
            'response': f"Error: {str(e)}",
            'model_used': llm_model,
            'embedding_model': embedding_model_name,
            'cross_encoder': cross_encoder_name
        }

def check_gpu():
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        memory_info = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        st.sidebar.success(f"✅ Using GPU for embeddings\n{gpu_info}\n{memory_info}")
    else:
        st.sidebar.warning("⚠️ GPU not detected. Using CPU for embeddings (may be slower)")



# Add this function to get ticker symbol from company name
def get_ticker_symbol(company_name, csv_path="fortune500_tickers.csv"):
    """
    Find ticker symbol for a company with fuzzy matching
    """
    try:
        # Read CSV file containing company names and tickers
        df = pd.read_csv(csv_path)
        
        # Use fuzzy matching to find closest company name
        matches = process.extractOne(company_name, df['Company'].tolist())
        
        # If match score is >= 75, return corresponding ticker
        if matches[1] >= 75:
            matched_company = matches[0]
            ticker = df[df['Company'] == matched_company]['Ticker'].values[0]
            return ticker, matched_company
        else:
            return None, None
    except Exception as e:
        st.error(f"Error finding ticker: {str(e)}")
        return None, None

# Add this function to get company information from Yahoo Finance
def get_company_info(ticker):
    """
    Get detailed company information using Yahoo Finance API
    """
    try:
        # Get company information
        company = yf.Ticker(ticker)
        info = company.info
        
        # Basic information
        company_info = {
            "name": info.get("longName", "N/A"),
            "ticker": ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "website": info.get("website", "N/A"),
            "logo_url": info.get("logo_url", None),
            
            # Market data
            "current_price": info.get("currentPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "open_price": info.get("open", "N/A"),
            "volume": info.get("volume", "N/A"),
            "day_high": info.get("dayHigh", "N/A"),
            "day_low": info.get("dayLow", "N/A"),
            "prev_close": info.get("previousClose", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"),
            
            # Valuation metrics
            "pe_ratio": info.get("trailingPE", "N/A"),
            "pb_ratio": info.get("priceToBook", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
            "enterprise_value": info.get("enterpriseValue", "N/A"),
            "ebitda": info.get("ebitda", "N/A"),
            
            # Market sentiment
            "recommendation": info.get("recommendationKey", "N/A"),
            "target_price": info.get("targetMeanPrice", "N/A"),
            "analyst_count": info.get("numberOfAnalystOpinions", "N/A"),
            
            # Business summary
            "business_summary": info.get("longBusinessSummary", "N/A"),
        }
        
        # Get news
        try:
            news = company.news
            company_info["news"] = news[:5] if news else []
        except:
            company_info["news"] = []
            
        # Get executives
        try:
            company_officers = info.get("companyOfficers", [])
            company_info["executives"] = company_officers
        except:
            company_info["executives"] = []
            
        return company_info
    except Exception as e:
        st.error(f"Error getting company information: {str(e)}")
        return None

# Add this function to display company information
def display_company_info(company_info):
    """
    Display company information in a formatted way
    """
    if not company_info:
        st.error("Unable to retrieve company information.")
        return
    
    # Create three columns for the header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    # with col2:
    #     st.title(f"{company_info['name']} ({company_info['ticker']})")
    #     st.caption(f"{company_info['sector']} | {company_info['industry']}")
    #     if company_info['website'] != "N/A":
    #         st.markdown(f"[Company Website]({company_info['website']})")
    
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
                {company_info['name']} ({company_info['ticker']})
            </div>
            <div style="font-size: 16px; color: black; text-align: center;">
                <strong>Sector:</strong> {company_info['sector']} | <strong>Industry:</strong> {company_info['industry']}
                {' | <strong>Website:</strong> <a href="' + company_info['website'] + '" target="_blank">Link</a>' if company_info['website'] != 'N/A' else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("***")
    
    # Create expandable section for detailed information
    with st.expander("View Detailed Company Information"):
        # Create tabs for different sections (excluding News)
        tabs = st.tabs(["Overview", "Market Data", "Valuation", "Sentiment", "Executives"])
        
        # Overview tab
        with tabs[0]:
            st.subheader("Business Summary")
            st.markdown(company_info['business_summary'])
        
        # Market data tab
        with tabs[1]:
            st.subheader("Market Data")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"${company_info['current_price']}")
                st.metric("Market Cap", f"${format_large_number(company_info['market_cap'])}")
                st.metric("Open", f"${company_info['open_price']}")
                st.metric("Volume", f"{format_large_number(company_info['volume'])}")
            
            with col2:
                st.metric("Day High", f"${company_info['day_high']}")
                st.metric("Day Low", f"${company_info['day_low']}")
                st.metric("Previous Close", f"${company_info['prev_close']}")
                st.metric("Employees", f"{format_large_number(company_info['employees'])}")
        
        # Valuation tab
        with tabs[2]:
            st.subheader("Valuation Metrics")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("P/E Ratio", f"{company_info['pe_ratio']}")
                st.metric("P/B Ratio", f"{company_info['pb_ratio']}")
            
            with col2:
                st.metric("ROE", f"{format_percentage(company_info['roe'])}")
                st.metric("Enterprise Value", f"${format_large_number(company_info['enterprise_value'])}")
                st.metric("EBITDA", f"${format_large_number(company_info['ebitda'])}")
        
        # Sentiment tab
        with tabs[3]:
            st.subheader("Market Sentiment")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Analyst Recommendation", f"{company_info['recommendation'].capitalize()}")
            
            with col2:
                st.metric("Target Price", f"${company_info['target_price']}")
                st.metric("Analyst Count", f"{company_info['analyst_count']}")
        
        # Executives tab
        with tabs[4]:
            st.subheader("Key Executives")
            
            if company_info["executives"]:
                for exec in company_info["executives"]:
                    with st.container():
                        st.markdown(f"### {exec.get('name', 'N/A')}")
                        st.caption(f"**Position:** {exec.get('title', 'N/A')}")
                        if exec.get('age'):
                            st.text(f"Age: {exec.get('age')}")
                        if exec.get('totalPay'):
                            st.text(f"Total Compensation: ${format_large_number(exec.get('totalPay'))}")
                        st.markdown("---")
            else:
                st.info("No executive information available.")

# Helper functions for formatting
def format_large_number(num):
    """Format large numbers for display"""
    if not num or num == "N/A":
        return "N/A"
    
    try:
        num = float(num)
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num/1_000:.2f}K"
        else:
            return f"{num:,.2f}"
    except:
        return str(num)

def format_percentage(value):
    """Format percentage values"""
    if not value or value == "N/A":
        return "N/A"
    
    try:
        return f"{float(value) * 100:.2f}%"
    except:
        return str(value)

def format_date(timestamp):
    """Format Unix timestamp to readable date"""
    if not timestamp:
        return "N/A"
    
    try:
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime('%B %d, %Y')
    except:
        return str(timestamp)

# Import Langchain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document as LangchainDocument
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def langchain_process_file(uploaded_file, company_name, embedding_model_name='all-mpnet-base-v2', chunking_strategy='semantic', chunk_size=1000):
    """
    Process uploaded file and create Langchain-based search indices with configurable embedding model, chunking strategy, and chunk size
    
    Args:
        uploaded_file: The uploaded file to process
        company_name: Name of the company
        embedding_model_name: Name of the embedding model to use (default: 'all-mpnet-base-v2')
        chunking_strategy: Strategy to use for text chunking ('semantic', 'fixed', or 'tfidf')
        chunk_size: Size of chunks to use (default: 1000)
    
    Returns:
        dict: Processing metrics including time statistics
    """
    try:
        # Start tracking processing time
        start_time = time.time()
        processing_metrics = {}
        
        # Create output directory for Langchain data
        output_dir = f"Company 10-K's/{company_name}/langchain"
        os.makedirs(output_dir, exist_ok=True)
        
        # Record PDF extraction start time
        extraction_start = time.time()
        
        # Process PDF with warning about Java if it fails
        try:
            text, tables = parse_10k_pdf(uploaded_file)
        except Exception as e:
            if "java" in str(e).lower():
                # Fallback to text-only extraction
                text = extract_text_only(uploaded_file)
                tables = []
            else:
                raise e
        
        # Record PDF extraction time
        extraction_time = time.time() - extraction_start
        processing_metrics['extraction_time'] = extraction_time
        
        # Record chunking start time
        chunking_start = time.time()
        
        # Determine chunk overlap based on the chunking strategy
        chunk_overlap = 200 if chunking_strategy == 'semantic' else 0
        
        # Use the appropriate chunking method for LangChain
        if chunking_strategy == 'semantic':
            # For semantic chunking, we'll use RecursiveCharacterTextSplitter with overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
            raw_documents = text_splitter.create_documents([text])
            
        elif chunking_strategy == 'fixed':
            # For fixed-length chunking, use CharacterTextSplitter without overlap
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                separator="\n\n"
            )
            raw_documents = text_splitter.create_documents([text])
            
        elif chunking_strategy == 'tfidf':
            # For TF-IDF based chunking, first split into potential chunks then apply TF-IDF logic
            # We'll use a combination of paragraph splitting and TF-IDF scoring in a custom function
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # For TF-IDF based chunking in LangChain, we need to approximate it
            # Since LangChain doesn't have a direct TF-IDF text splitter,
            # we'll use TextTilingTokenizer from NLTK as an approximation
            try:
                from nltk.tokenize.texttiling import TextTilingTokenizer
                tt = TextTilingTokenizer()
                # TextTiling works better with longer texts
                tiled_text = tt.tokenize(text)
                
                # Create documents from the tiled sections
                raw_documents = [Document(page_content=tile.strip()) for tile in tiled_text if tile.strip()]
                
                # If TextTiling produces too few chunks, fall back to paragraph splitting
                if len(raw_documents) < 5:
                    text_splitter = CharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap//2,  # Use less overlap for tfidf approximation
                        separator="\n\n"
                    )
                    raw_documents = text_splitter.create_documents([text])
            except:
                # Fallback if NLTK is not available
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap//2,
                    separator="\n\n"
                )
                raw_documents = text_splitter.create_documents([text])
        else:
            # Default to semantic chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
            raw_documents = text_splitter.create_documents([text])
        
        # Record chunking time
        chunking_time = time.time() - chunking_start
        processing_metrics['chunking_time'] = chunking_time
        
        # Process tables - convert to LangChain documents
        table_documents = []
        if tables:
            for i, table in enumerate(tables):
                # Convert table to string representation
                table_text = table.to_string(index=False)
                table_documents.append(Document(
                    page_content=table_text,
                    metadata={"type": "table", "table_id": i}
                ))
        
        # Combine text and table documents
        all_documents = raw_documents + table_documents
        
        # Add section metadata to documents
        documents = []
        for i, doc in enumerate(all_documents):
            # Try to identify section titles
            content = doc.page_content
            section_title = "General"
            # Check if the document starts with what looks like a section title
            lines = content.split('\n')
            if lines and (re.match(r'^[A-Z\s]{5,}$', lines[0]) or re.match(r'^ITEM\s+\d+', lines[0], re.IGNORECASE)):
                section_title = lines[0]
            
            # Create metadata
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            metadata.update({
                'chunk_id': i,
                'section_title': section_title,
                'type': metadata.get('type', 'text')
            })
            
            # Create new document with updated metadata
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        # Record embedding start time
        embedding_start = time.time()
        
        # Initialize embedding model based on selection
        if embedding_model_name == 'all-mpnet-base-v2':
            embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
        elif embedding_model_name == 'yiyanghkust/finbert-tone':
            embeddings = HuggingFaceEmbeddings(model_name='yiyanghkust/finbert-tone')
        else:
            # Default to all-mpnet-base-v2
            embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
        
        # Create Langchain vector store
        vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5  # Number of documents to return
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(search_kwargs={"k": 5}),
                bm25_retriever
            ],
            weights=[0.5, 0.5]
        )
        
        # Record embedding and indexing time
        embedding_time = time.time() - embedding_start
        processing_metrics['embedding_time'] = embedding_time
        
        # Save vector store to disk
        vectorstore.save_local(os.path.join(output_dir, f"{company_name.lower()}_vectorstore"))
        
        # Save document objects for future reference
        with open(os.path.join(output_dir, f"{company_name.lower()}_documents.pkl"), "wb") as f:
            pickle.dump(documents, f)
        
        # Additionally, save text content of documents as JSON for easier inspection
        docs_json = []
        for i, doc in enumerate(documents):
            docs_json.append({
                "id": i,
                "text": doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(os.path.join(output_dir, f"{company_name.lower()}_documents.json"), "w") as f:
            json.dump(docs_json, f)
        
        # Save chunking metadata including processing time statistics
        chunking_metadata = {
            'strategy': chunking_strategy,
            'embedding_model': embedding_model_name,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'num_chunks': len(documents),
            'processing_metrics': processing_metrics,
            'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_processing_time': time.time() - start_time
        }
        
        with open(os.path.join(output_dir, f"{company_name.lower()}_langchain_metadata.json"), "w") as f:
            json.dump(chunking_metadata, f)
        
        # Return processing metrics
        return {
            'success': True,
            'metrics': processing_metrics,
            'total_time': time.time() - start_time,
            'num_chunks': len(documents)
        }
        
    except Exception as e:
        st.error(f"Error processing file with Langchain: {str(e)}")
        raise e

# Also update the langchain_rag_pipeline to support model selection
def langchain_rag_pipeline(query, base_path, company_name, llm_model="mistral", embedding_model_name=None):
    """
    Simple Langchain RAG pipeline that avoids pickle deserialization with model selection
    
    Args:
        query: User query
        base_path: Base path to the company data
        company_name: Name of the company
        llm_model: LLM to use for generation ("mistral" or "gemini")
        embedding_model_name: Optional - embedding model to use (will use stored model if None)
    """
    try:
        # Set up the langchain directory path
        langchain_path = f"{base_path}/langchain"
        
        # Check if model_info.json exists and load it
        if embedding_model_name is None and os.path.exists(f"{langchain_path}/model_info.json"):
            with open(f"{langchain_path}/model_info.json", "r") as f:
                model_info = json.load(f)
                embedding_model_name = model_info.get("embedding_model", "all-mpnet-base-v2")
        elif embedding_model_name is None:
            # Default if no model info is stored
            embedding_model_name = "all-mpnet-base-v2"
        
        # Load the embeddings model
        embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Load the documents from JSON
        with open(f"{langchain_path}/documents.json", "r") as f:
            doc_dicts = json.load(f)
            docs = [LangchainDocument(page_content=d["page_content"], metadata=d["metadata"]) 
                   for d in doc_dicts]
        
        # Load the embeddings from JSON
        with open(f"{langchain_path}/embeddings.json", "r") as f:
            stored_embeddings = json.load(f)
            
        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 4
            
        # Create a semantic search function using the pre-computed embeddings
        query_embedding = embeddings_model.embed_query(query)
        
        # Simple vector similarity search using cosine similarity
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a*b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a*a for a in vec1) ** 0.5
            magnitude2 = sum(b*b for b in vec2) ** 0.5
            return dot_product / (magnitude1 * magnitude2)
        
        # Calculate similarity scores for all documents
        similarities = [(i, cosine_similarity(query_embedding, doc_embedding)) 
                       for i, doc_embedding in enumerate(stored_embeddings)]
        
        # Sort by similarity and get top 4
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:4]]
        
        # Get the top documents from vector search
        vector_docs = [docs[i] for i in top_indices]
        
        # Get documents from BM25
        keyword_docs = bm25_retriever.get_relevant_documents(query)
        
        # Combine results (simple ensemble method)
        all_docs = vector_docs + keyword_docs
        seen_content = set()
        unique_docs = []
        
        # Remove duplicates while preserving order
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        # Get final top docs (limit to 4)
        retrieved_docs = unique_docs[:4]
        
        # Format contexts
        contexts = []
        for i, doc in enumerate(retrieved_docs[:2]):  # Only use top 2 docs
            contexts.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': 1.0 - (i * 0.1),  # Simple scoring
                'prev_text': '',
                'next_text': ''
            })
        
        # Format the context for the selected model
        formatted_context = format_retrieved_context(contexts)
        
        # Generate response using selected model
        if llm_model.lower() == "gemini":
            response = generate_gemini_response(query, formatted_context, company_name)
        else:  # Default to Mistral
            response = generate_mistral_response(query, formatted_context, MISTRAL_API_KEY, company_name)
        
        return {
            'query': query,
            'retrieved_results': contexts,
            'top_2_contexts': formatted_context.split('\n\n'),
            'response': response,
            'model_used': llm_model,
            'embedding_model': embedding_model_name
        }
    except Exception as e:
        st.error(f"Error in Langchain RAG pipeline: {str(e)}")
        return {
            'query': query,
            'retrieved_results': [],
            'top_2_contexts': [],
            'response': f"Error: {str(e)}",
            'model_used': llm_model,
            'embedding_model': embedding_model_name
        }
    
# Finally, modify the main function to include the LLM selection option
# Add this inside the document processing options section, alongside the chunking strategy and embedding model selections

def main():
    st.markdown("<h3 style='text-align: center;'>10-K Report Analysis</h3>", unsafe_allow_html=True)
    st.markdown("""
    This application allows you to extract information from 10-K reports and ask questions about them.
    Upload your 10-K PDF file to get started, then ask questions about the document.
    """)
    
    # Center-align the "Upload New Document" button using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Upload New Document", use_container_width=True):
            st.session_state.document_processed = False
            st.session_state.current_company = None
            st.rerun()
    
    # Add company information section
    st.markdown("### Quick Company Analysis")
    st.markdown("Enter your company name to get a general overview without uploading a 10-K")
    
    # Company name input for Yahoo Finance search
    company_search = st.text_input("Company name for analysis:", placeholder="e.g., Nike, Apple, Microsoft")
    
    # When a company name is entered, fetch and display information
    if company_search:
        with st.spinner("Fetching company information..."):
            # Get ticker symbol
            ticker, matched_company = get_ticker_symbol(company_search)
            
            if ticker:
                # Fetch company information
                company_info = get_company_info(ticker)
                
                # Display company information
                if company_info:
                    display_company_info(company_info)
                else:
                    st.error("Could not retrieve company information.")
            else:
                st.error(f"Could not find a matching ticker symbol for '{company_search}'.")
                st.info("Try a different company name or check spelling.")
    
    # Sidebar info
    st.sidebar.title("About This App")
    st.sidebar.markdown("""
    This application uses advanced RAG techniques to analyze 10-K reports:
    * PDF parsing with text and table extraction
    * Multiple text chunking strategies
    * Hybrid search (BM25 + FAISS)
    * Maximum Marginal Relevance for diversity
    * Various Cross-encoder reranking options
    * Multiple LLM options (Mistral AI and Google Gemini)
    * Multiple embedding models (general and financial)
    * Yahoo Finance integration for quick company analysis
    * LangChain integration option
    """)
    
    # Instructions in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Instructions:**
    1. (Optional) Enter Company Name for a quick Financial Analysis
    2. Select text chunking strategy, chunk size and embedding model
    3. Upload your 10-K PDF file
    4. Wait for processing to complete
    5. Choose your preferred implementation approach, LLM, and Cross-Encoder Model
    6. Ask questions about the document
    """)

    # Check for GPU
    check_gpu()
    
    # Session state for processed company info
    if 'processed_companies' not in st.session_state:
        st.session_state.processed_companies = []
        # Try to find existing processed companies
        try:
            company_dirs = [d for d in os.listdir("Company 10-K's") if os.path.isdir(os.path.join("Company 10-K's", d))]
            st.session_state.processed_companies = company_dirs
        except FileNotFoundError:
            pass
    
    # Track if a document is currently being processed
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
        
    # Track the current company being processed
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    
    # Track the current query
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Track selected embedding model
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = 'all-mpnet-base-v2'
        
    # Track selected chunking strategy
    if 'chunking_strategy' not in st.session_state:
        st.session_state.chunking_strategy = 'semantic'
    
    # Track selected chunk size
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 1024
        
    # Track selected cross-encoder model
    if 'cross_encoder' not in st.session_state:
        st.session_state.cross_encoder = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    # Function to set the query when an example is clicked
    def set_query(query_text):
        st.session_state.current_query = query_text
    
    # Only show the query section after document processing
    if st.session_state.document_processed:
        st.success("Document processing done, you can now enter your query.")
        st.markdown("---")
        st.subheader("Ask questions about the 10-K")
        
        # Company selection dropdown
        selected_company = st.selectbox(
            "Document Uploaded:", 
            st.session_state.processed_companies,
            index=st.session_state.processed_companies.index(st.session_state.current_company) if st.session_state.current_company in st.session_state.processed_companies else 0
        )
        
        # Load chunking metadata if available
        chunking_metadata = {}
        try:
            metadata_path = f"Company 10-K's/{selected_company}/{selected_company.lower()}_chunking_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    chunking_metadata = json.load(f)
        except:
            chunking_metadata = {}
        
        # Display chunking strategy, chunk size, embedding model used, and processing time
        if chunking_metadata:
            chunking_info = chunking_metadata.get('strategy', 'semantic')
            embedding_info = chunking_metadata.get('embedding_model', st.session_state.embedding_model)
            chunk_size_info = chunking_metadata.get('chunk_size', st.session_state.chunk_size)
            processing_time = chunking_metadata.get('processing_time', 'Not recorded')
            
            st.info(f"Document was processed using {chunking_info} chunking strategy with {chunk_size_info} chunk size and {embedding_info} embedding model. Processing time: {processing_time} seconds")
        else:
            st.info(f"Document was processed with {st.session_state.embedding_model} embedding model")
        
        # Two columns for implementation and LLM selection
        col1, col2 = st.columns(2)
        
        with col1:
            # RAG Implementation selection
            st.markdown("### RAG Implementation")
            rag_implementation = st.radio(
                "Select RAG Implementation:",
                ["Custom RAG Implementation", "Langchain Implementation"],
                horizontal=True
            )
            
            # Convert selection to boolean
            use_langchain = (rag_implementation == "Langchain Implementation")
        
        with col2:
            # LLM Model selection
            st.markdown("### LLM Selection")
            llm_model = st.radio(
                "Select LLM:",
                ["Mistral", "Gemini"],
                horizontal=True
            )
        
        # Cross-encoder model selection
        st.markdown("### Cross-Encoder Model")
        cross_encoder_options = {
            "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM-L-6 (Fast)",
            "cross-encoder/ms-marco-electra-base": "ELECTRA-Base (Balanced)",
            "cross-encoder/stsb-roberta-base": "RoBERTa-Base (High Quality)"
        }
        
        # Only show cross-encoder selection for custom implementation
        if not use_langchain:
            cross_encoder_model = st.radio(
                "Select Cross-Encoder Model for Reranking:",
                options=list(cross_encoder_options.keys()),
                format_func=lambda x: cross_encoder_options[x],
                horizontal=True
            )
            
            st.session_state.cross_encoder = cross_encoder_model
            
            # Display explanation based on selection
            if cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2":
                st.info("MiniLM-L-6: Fastest model with good performance for most scenarios.")
            elif cross_encoder_model == "cross-encoder/ms-marco-electra-base":
                st.info("ELECTRA-Base: Good balance between speed and quality for financial documents.")
            else:
                st.info("RoBERTa-Base: Highest quality but slower performance. Best for complex financial analysis.")
        else:
            st.info("Cross-encoder selection is only available for Custom RAG Implementation.")
            cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default for Langchain
        
        # Add informative messages about which implementation is being used
        implementation_info = "Using Langchain implementation with ensemble retrieval" if use_langchain else f"Using custom RAG implementation with hybrid search, MMR, and {cross_encoder_options[cross_encoder_model]} reranking"
        model_info = f"Responses will be generated using {llm_model} API"
        
        st.info(f"{implementation_info}. {model_info}.")
        
        # Advanced options expander
        with st.expander("Advanced Options"):
            # Lambda parameter (only for custom implementation)
            if not use_langchain:
                lambda_param = st.slider(
                    "Semantic vs. Keyword Balance:", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.5,
                    help="Higher values favor semantic search, lower values favor keyword search")
            else:
                lambda_param = 0.5  # Default value for Langchain implementation
                st.info("Advanced options are not available for Langchain implementation.")
        
        # Example queries section
        st.markdown("### Example Queries")
        example_queries = [
            "What are the main revenue sources?",
            "Describe the strategy for international markets",
            "What are the key financial risks?",
            "How does the company approach sustainability?",
            "What are the major competitors mentioned?"
        ]
        
        # Display example queries as buttons in rows of 2 or 3
        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(query, key=f"example_query_{i}"):
                    set_query(query)
                    
        st.markdown("---")
        
        # Query input with value from session state
        query = st.text_area("Enter your query:", 
                            value=st.session_state.current_query,
                            height=100,
                            placeholder="E.g., What are the major risk factors mentioned in the report?")
        
        # Submit button
        submit_button = st.button("Submit Query")
        
        # Results section
        if submit_button and query:
            # Clear the current query after submission
            st.session_state.current_query = ""
            
            st.markdown("---")
            
            implementation_name = "Langchain" if use_langchain else "Custom"
            embedding_model = st.session_state.embedding_model
            with st.spinner(f"{implementation_name} RAG is decoding financial insights using {embedding_model} embeddings and {llm_model}, please hold on....."):
                # Set up base path
                base_path = f"Company 10-K's/{selected_company}"
                
                # Execute the appropriate RAG pipeline based on user selection
                if use_langchain:
                    result = langchain_rag_pipeline(
                        query, 
                        base_path, 
                        selected_company,
                        llm_model=llm_model.lower(),
                        embedding_model_name=embedding_model
                    )
                    
                    # Add post-processing for responses to fix formatting issues
                    if "response" in result:
                        # Clean up formatting issues in the response
                        response = result["response"]
                        # Fix the common formatting issues with numbers and spacing
                        response = re.sub(r'(\d+)million', r'\1 million', response)
                        response = re.sub(r'(\d+)\*million\*', r'\1 million', response)
                        response = re.sub(r'withanoperatingincomeof', r'with an operating income of ', response)
                        response = re.sub(r'\*withanoperatingincomeof\*', r'with an operating income of ', response)
                        response = re.sub(r'millionandanoperatingincomeof', r'million and an operating income of ', response)
                        response = re.sub(r'\*millionandanoperatingincomeof\*', r'million and an operating income of ', response)
                        # Update the result with the cleaned response
                        result["response"] = response
                else:
                    result = rag_pipeline(
                        query, 
                        base_path, 
                        selected_company,
                        lambda_param=lambda_param,
                        section_filter=None,
                        llm_model=llm_model.lower(),
                        embedding_model_name=embedding_model,
                        cross_encoder_name=cross_encoder_model
                    )
                
                # Display results
                st.markdown(f"## Response (Generated by {result.get('model_used', llm_model)})")
                st.info(result["response"])
                
                # Display models used
                embedding_used = result.get('embedding_model', embedding_model)
                cross_encoder_used = result.get('cross_encoder', cross_encoder_model)
                
                # Only show cross-encoder info for custom implementation
                if not use_langchain:
                    st.caption(f"Embedding model: {embedding_used} | Cross-encoder: {cross_encoder_options.get(cross_encoder_used, cross_encoder_used)}")
                else:
                    st.caption(f"Embedding model: {embedding_used}")
                
                # Show retrieved contexts in an expander
                with st.expander("View Retrieved Contexts"):
                    for i, context in enumerate(result["retrieved_results"], 1):
                        st.markdown(f"**Context {i}** (Score: {context.get('score', 'N/A')})")
                        if 'metadata' in context:
                            st.markdown(f"Section: {context['metadata'].get('section_title', 'General')}")
                        st.text(context['text'][:500] + "..." if len(context['text']) > 500 else context['text'])
                        st.markdown("---")
    else:
        # Add chunking strategy selection before embedding model selection
        st.subheader("Document Processing Options")
        
        # Create explanations for each chunking strategy
        chunking_explanations = {
            "semantic": "Semantic chunking preserves meaning and context by creating chunks based on content relationships. Best for maintaining contextual coherence.",
            "fixed": "Fixed-length chunking creates uniform, non-overlapping chunks of consistent size. Simple and fast but may break contextual relationships.",
            "tfidf": "TF-IDF based chunking uses term importance to determine chunk boundaries, keeping related content together. Good for documents with distinct topical sections."
        }
        
        # Create three columns for chunking, chunk size, and embedding selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Chunking strategy selection
            st.markdown("### Text Chunking Strategy")
            chunking_options = {
                "semantic": "Semantic Chunking (with overlap)",
                "fixed": "Fixed-Length Chunking (non-overlapping)",
                "tfidf": "TF-IDF Based Chunking"
            }
            
            selected_chunking = st.radio(
                "Select Text Chunking Strategy:",
                options=list(chunking_options.keys()),
                format_func=lambda x: chunking_options[x],
                help="Choose how the document will be split into chunks for processing"
            )
            
            st.session_state.chunking_strategy = selected_chunking
            
            # Display explanation for selected chunking strategy
            st.info(chunking_explanations[selected_chunking])
        
        with col2:
            # Chunk size selection
            st.markdown("### Chunk Size")
            chunk_size_options = {
                256: "Small (256 tokens)",
                512: "Medium (512 tokens)",
                1024: "Large (1024 tokens)"
            }
            
            selected_chunk_size = st.radio(
                "Select Chunk Size:",
                options=list(chunk_size_options.keys()),
                format_func=lambda x: chunk_size_options[x],
                help="Choose the maximum size of each text chunk (smaller chunks may improve retrieval accuracy but create more chunks)"
            )
            
            st.session_state.chunk_size = selected_chunk_size
            
            # Display explanation for selected chunk size
            if selected_chunk_size == 256:
                st.info("Small chunks are good for precise retrieval but may lose broader context. Processing will be slower due to more chunks.")
            elif selected_chunk_size == 512:
                st.info("Medium chunks balance precision and context. Good general-purpose size for most documents.")
            else:
                st.info("Large chunks preserve more context but may reduce precision for specific queries. Processing will be faster with fewer chunks.")
        
        with col3:
            # Embedding model selection
            st.markdown("### Embedding Model")
            embedding_model_options = {
                "all-mpnet-base-v2": "General Purpose (all-mpnet-base-v2)",
                "yiyanghkust/finbert-tone": "Financial Domain (FinBERT)"
            }
            
            selected_embedding_model = st.radio(
                "Select Embedding Model:",
                options=list(embedding_model_options.keys()),
                format_func=lambda x: embedding_model_options[x],
                help="Choose between a general purpose embedding model or a financial domain-specific model"
            )
            
            st.session_state.embedding_model = selected_embedding_model
            
            # Display explanation for selected embedding model
            if selected_embedding_model == "all-mpnet-base-v2":
                st.info("General Purpose model works well for most text analysis tasks and provides good overall performance.")
            else:
                st.info("Financial Domain model (FinBERT) is specifically trained on financial texts and may provide better results for financial document analysis.")
        
        # Show comparison of chunking strategies in an expander
        with st.expander("Chunking Strategy Comparison"):
            st.markdown("""
            | Feature | Semantic Chunking | Fixed-Length Chunking | TF-IDF Based Chunking |
            |---------|-------------------|------------------------|------------------------|
            | Method | Preserves content relationships | Splits text into uniform segments | Uses term importance to determine boundaries |
            | Overlap | Yes (configurable) | No | No (boundary-based) |
            | Best for | Complex documents with varied content | Simple documents with uniform structure | Documents with distinct topical sections |
            | Pros | Maintains context between related concepts | Simple, predictable chunk sizes | Adapts to content meaning and structure |
            | Cons | More complex, may create uneven chunks | May break meaningful content | Performance depends on vocabulary distribution |
            """)
        
        # File uploader section only shown when no document is being processed
        uploaded_file = st.file_uploader("Upload 10-K PDF", type="pdf")
        
        if uploaded_file is not None:
            # Company name input
            company_name = st.text_input("Document Uploaded:",  
                                       value=uploaded_file.name.split('.')[0].upper())
            
            # Auto-process the file when name is entered
            if company_name:
                chunking_strategy = st.session_state.chunking_strategy
                embedding_model = st.session_state.embedding_model
                chunk_size = st.session_state.chunk_size
                
                # Display AI facts outside the spinner
                ai_facts = [
                    "Did you know? The term 'Natural Language Processing' was coined in the 1950s, but humans have been fascinated with language-understanding machines since the 1600s!",
                    "Fun fact: Modern NLP systems can analyze sentiment in over 100 languages, but still struggle with sarcasm... just like some humans!",
                    "While we work: GPT models are trained on trillions of tokens, equivalent to reading thousands of encyclopedias worth of text.",
                    "AI Tidbit: The average NLP model today processes more text in a second than a human could read in an entire year!",
                    "NLP Fact: The first chatbot ELIZA was created in 1966 at MIT and could simulate conversation by pattern matching.",
                    "Finance AI Fact: FinBERT is a specialized language model pre-trained on financial communications to better understand financial terminology.",
                    f"Processing Insight: {chunking_options[chunking_strategy]} is particularly effective for financial documents because it {chunking_explanations[chunking_strategy].split('.')[0].lower()}."
                ]
                fact_placeholder = st.empty()
                fact_placeholder.info(random.choice(ai_facts))
                
                # Use st.spinner to display a spinner during processing
                with st.spinner(f"Processing document with {chunking_options[chunking_strategy]}, {chunk_size_options[chunk_size]} chunk size, and {embedding_model_options[embedding_model]} embedding model..."):
                    # Initialize timer
                    start_time = time.time()
                    
                    try:
                        # Process with your original method, passing the selected embedding model, chunking strategy, and chunk size
                        process_file(
                            uploaded_file, 
                            company_name, 
                            embedding_model_name=embedding_model,
                            chunking_strategy=chunking_strategy,
                            chunk_size=chunk_size
                        )
                        
                        # Update langchain_process_file to also accept chunking_strategy and chunk_size parameters
                        langchain_process_file(
                            uploaded_file, 
                            company_name, 
                            embedding_model_name=embedding_model,
                            chunking_strategy=chunking_strategy,
                            chunk_size=chunk_size
                        )
                        
                        # Calculate total processing time
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Update the chunking metadata with processing time
                        output_dir = f"Company 10-K's/{company_name}"
                        try:
                            metadata_path = os.path.join(output_dir, f"{company_name.lower()}_chunking_metadata.json")
                            if os.path.exists(metadata_path):
                                with open(metadata_path, "r") as f:
                                    chunking_metadata = json.load(f)
                                
                                chunking_metadata['processing_time'] = round(processing_time, 2)
                                chunking_metadata['chunk_size'] = chunk_size
                                
                                with open(metadata_path, "w") as f:
                                    json.dump(chunking_metadata, f)
                        except Exception as e:
                            st.warning(f"Could not update metadata with processing time: {str(e)}")
                        
                        # Update the session state
                        if company_name not in st.session_state.processed_companies:
                            st.session_state.processed_companies.append(company_name)
                        
                        st.session_state.document_processed = True
                        st.session_state.current_company = company_name
                        
                        # Show success message with processing time
                        st.success(f"Document processed successfully in {processing_time:.2f} seconds!")
                        
                        # Wait a moment to show the success message
                        time.sleep(2)
                        
                        # Use the current Streamlit rerun method 
                        st.rerun()
                        
                    except Exception as e:
                        # Show error message
                        st.error(f"Error processing document: {str(e)}")
        else:
            st.info("Please upload a 10-K report PDF to get started.")

def process_file(uploaded_file, company_name, embedding_model_name='all-mpnet-base-v2', chunking_strategy='semantic', chunk_size=1024):
    """
    Process uploaded file and create search indices with configurable embedding model and chunking strategy
    
    Args:
        uploaded_file: The uploaded file to process
        company_name: Name of the company
        embedding_model_name: Name of the embedding model to use (default: 'all-mpnet-base-v2')
        chunking_strategy: Strategy to use for text chunking ('semantic', 'fixed', or 'tfidf')
        chunk_size: Size of each chunk (default: 1024)
    """
    try:
        # Record start time for processing
        start_time = time.time()
        
        # Create output directory
        output_dir = f"Company 10-K's/{company_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process PDF with warning about Java if it fails
        try:
            text, tables = parse_10k_pdf(uploaded_file)
        except Exception as e:
            if "java" in str(e).lower():
                # Fallback to text-only extraction
                text = extract_text_only(uploaded_file)
                tables = []
            else:
                raise e
        
        # Calculate overlap based on chunk size (for semantic chunking)
        overlap = int(chunk_size * 0.2)  # 20% overlap
        
        # Apply the selected chunking strategy
        if chunking_strategy == 'semantic':
            text_chunks = semantic_chunk_text(text, max_chunk_size=chunk_size, overlap=overlap)
        elif chunking_strategy == 'fixed':
            text_chunks = fixed_length_chunk_text(text, chunk_size=chunk_size)
        elif chunking_strategy == 'tfidf':
            text_chunks = tfidf_chunk_text(text, max_chunk_size=chunk_size, min_chunk_size=int(chunk_size * 0.2))
        else:
            # Default to semantic chunking
            text_chunks = semantic_chunk_text(text, max_chunk_size=chunk_size, overlap=overlap)
        
        # Process tables (this doesn't change with chunking strategy)
        table_chunks = tables_to_text(tables, base_chunk_id=len(text_chunks)) if tables else []
        
        # Combine text and table chunks
        all_chunks = text_chunks + table_chunks
        
        # Extract sections
        all_chunks = extract_sections(all_chunks)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save chunking metadata
        chunking_metadata = {
            'strategy': chunking_strategy,
            'embedding_model': embedding_model_name,
            'chunk_size': chunk_size,
            'num_chunks': len(all_chunks),
            'date_processed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_time': round(processing_time, 2)
        }
        
        with open(os.path.join(output_dir, f"{company_name.lower()}_chunking_metadata.json"), "w") as f:
            json.dump(chunking_metadata, f)
        
        # Create search index with the selected embedding model
        faiss_index, bm25, embedding_model, *_ = create_search_index(all_chunks, company_name, embedding_model_name=embedding_model_name)
        
        # Save components
        save_search_index(faiss_index, all_chunks, bm25, output_dir, company_name)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise e

# Add this function as a fallback when Java is not available
def extract_text_only(pdf_file):
    """Extract text from PDF without using Java-based table extraction"""
    import io
    from PyPDF2 import PdfReader
    
    # Create a PDF reader object
    pdf_bytes = io.BytesIO(pdf_file.getvalue())
    reader = PdfReader(pdf_bytes)
    
    # Extract text from all pages
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
    return text

if __name__ == "__main__":
    # Set page title and layout
    st.set_page_config(page_title="10-K Report Analysis", layout="wide")
    # Check if the app is running for the first time and create necessary directories
    if not os.path.exists("Company 10-K's"):
        os.makedirs("Company 10-K's", exist_ok=True)
    
    # Start the main application
    main()