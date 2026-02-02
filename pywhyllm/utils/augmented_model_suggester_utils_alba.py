import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import requests
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Optional


def find_top_match_in_causenet(causenet_dict, variable1, variable2, threshold=0.35):
    """
    Find the top matching causal pair in CauseNet for given variables.
    
    Args:
        causenet_dict: Dictionary of CauseNet causal relations
        variable1: First variable name
        variable2: Second variable name
        threshold: Similarity threshold (0.0-1.0). Lower = more permissive. Default: 0.35
                   (Lowered to capture semantic matches like "age-weight")
    
    Returns:
        Matching CauseNet entry or None if no match above threshold
    """
    # Helper function to normalize variable names (remove special chars, lowercase)
    def normalize_var(var):
        # Remove special characters and convert to lowercase
        import re
        normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', var.lower())
        # Split into words and remove common stopwords
        words = normalized.split()
        stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for'}
        return ' '.join([w for w in words if w not in stopwords and len(w) > 2])
    
    # Normalize input variables
    var1_norm = normalize_var(variable1)
    var2_norm = normalize_var(variable2)
    
    pair_strings = [
        f"{causenet_dict[key]['causal_relation']['cause']}-{causenet_dict[key]['causal_relation']['effect']}"
        for key in causenet_dict]

    tokenized_pairs = [text.split() for text in pair_strings]
    bm25 = BM25Okapi(tokenized_pairs)

    # Create queries with normalized variables (use hyphen like original CauseNet format)
    query = var1_norm + "-" + var2_norm
    reverse_query = var2_norm + "-" + var1_norm
    
    # Tokenize and combine both directions
    tokenized_query = query.split()
    tokenized_reverse_query = reverse_query.split()
    combined_query = list(set(tokenized_query + tokenized_reverse_query))

    # Increase k to get more candidates for semantic matching (was 5, now 50)
    k = min(50, len(pair_strings))  # Don't exceed total pairs
    scores = bm25.get_scores(combined_query)
    top_k_indices = np.argsort(scores)[::-1][:k]
    candidate_pairs = [pair_strings[i] for i in top_k_indices]

    # Use semantic similarity on expanded candidate set
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode original queries (not just normalized) for better semantic matching
    query_full = f"{variable1} {variable2}"
    reverse_query_full = f"{variable2} {variable1}"
    
    query_embedding = model.encode(query_full, convert_to_tensor=True)
    reverse_query_embedding = model.encode(reverse_query_full, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_pairs, convert_to_tensor=True)

    similarities = util.cos_sim(query_embedding, candidate_embeddings).flatten()
    reverse_similarities = util.cos_sim(reverse_query_embedding, candidate_embeddings).flatten()

    max_similarities = np.maximum(similarities, reverse_similarities)

    top_idx = np.argmax(max_similarities)
    top_similarity = max_similarities[top_idx]
    top_pair = candidate_pairs[top_idx]

    if top_similarity >= threshold:
        print(f"✓ CauseNet match: {top_pair} (Similarity: {top_similarity:.4f})")
        return causenet_dict[top_pair]
    else:
        print(f"✗ No CauseNet match above threshold {threshold} (Best: {top_similarity:.4f} for '{top_pair}')")
        return None


def get_source_text(causenet_query_result):
    source_text = ""
    if causenet_query_result:
        for item in causenet_query_result["sources"]:
            if item["type"] == 'wikipedia_sentence' or item["type"] == 'clueweb12_sentence':
                source_text += item["payload"]["sentence"] + " "

    return source_text

# This is done in create combined retriever

# def split_data_and_create_vectorstore_retriever(source_text):
#     document = Document(page_content=source_text)

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=100,
#         chunk_overlap=20
#     )
#     splits = text_splitter.split_documents([document])

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db"  # Optional: Save to disk for reuse
#     )

#     retriever = vectorstore.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 5}
#     )

#     return retriever


def query_llm(variable1, variable2, source_text=None, retriever=None, llm=None, return_prompt=False, confidence_level=False):
    """
    Query the LLM to determine causal relationship between two variables.
    
    Args:
        variable1: First variable name
        variable2: Second variable name
        source_text: Optional context text for RAG
        retriever: Optional retriever for RAG chain
        llm: Language model instance (if None, creates default OpenAI())
        return_prompt: If True, returns (response, system_prompt) tuple. Default: False
        confidence_level: If True, instructs LLM to provide confidence score. Default: False
    
    Returns:
        If return_prompt=False: String response from LLM
        If return_prompt=True: Tuple (response, system_prompt) where system_prompt 
                               is the complete system prompt sent to LLM with context filled
    """
    if llm is None:
        llm = OpenAI()
    
    if source_text:
        system_prompt = """You are a helpful assistant for causal reasoning.

    Context: {context}
    """
    else:
        system_prompt = """You are a helpful assistant for causal reasoning.
    """
    
    confidence_instruction = ""
    if confidence_level:
        confidence_instruction = """ 
Additionally, provide TWO scores within tags:
1. <confidence></confidence>: Your confidence in this causal judgment (0-1 scale)
   - How certain are you that you chose the correct relationship (A, B, or C)?
   - Example: High confidence = 0.9 (very sure), Low confidence = 0.5 (uncertain)
   - If the answer is C (no relationship), confidence reflects certainty of no relationship.

2. <strength></strength>: The strength of the causal relationship (0-1 scale)
   - IF a causal relationship exists (A or B), how strong/deterministic is it?
   - 1.0  = Very strong relationship (e.g., "Smoking → Lung Cancer")
   - 0.7 = Moderate relationship (e.g., "Age → Heart Attack" - depends on genetics, lifestyle)
   - 0.5 = Moderate-weak relationship (e.g., "Education → Income" - many exceptions)
   - 0.3 = Weak relationship (e.g., "Birth Order → Personality" - small effect, many confounders)
   - 0.0 = No causal relationship (if the answer is C)
   
Important distinctions:
- You can have HIGH confidence (0.9) that a WEAK relationship (0.3) exists
- You can have LOW confidence (0.5) about a potentially STRONG relationship (0.8)
- If the answer is C (no relationship), set strength to 0.0 but confidence reflects certainty of no relationship and confidence can be 1.0 (very sure no relationship exists)
- Strength reflects the SIZE of the causal effect, not your confidence in the judgment
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    query = f"""Determine the most likely cause-and-effect relationship between "{variable1}" and "{variable2}".

Carefully analyze the three options below and select ONE:

Option A: {variable1} → {variable2} (meaning: {variable1} CAUSES {variable2})
Option B: {variable2} → {variable1} (meaning: {variable2} CAUSES {variable1})
Option C: No causal relationship exists between them

IMPORTANT: 
- Think step-by-step about which variable influences the other
- Your reasoning should clearly match your final answer
- If you conclude "X causes Y", make sure you select the option where X is the cause

Provide your reasoning first, then give your final answer as ONLY the letter (A, B, or C) within <answer></answer> tags.{confidence_instruction}"""

    if source_text:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": query})
        
        if return_prompt:
            # Extract retrieved documents to see what context was actually sent
            retrieved_docs = response.get('context', [])
            retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Build the actual prompt that LangChain sent to the LLM
            actual_prompt = f"""You are a helpful assistant for causal reasoning.

    Context: {retrieved_context}
    """
            return response['answer'], actual_prompt
        else:
            return response['answer']

    else:
        default_chain = prompt | llm
        response = default_chain.invoke({"input": query})
        
        if return_prompt:
            # No RAG, return the basic system prompt
            return response, system_prompt
        else:
            return response

# PUBMED INTEGRATION FUNCTIONS

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def rewrite_query_for_pubmed(variable1: str, variable2: str) -> List[str]:
    """
    Create multiple query variations for PubMed search to maximize retrieval.
    
    Args:
        variable1: First variable name
        variable2: Second variable name
    
    Returns:
        List of query strings ordered by specificity (most specific first)
    """
    queries = [
        # Most specific - causal language
        f"{variable1} AND {variable2} AND (causal OR causation OR cause)",
        f"{variable1} AND {variable2} AND (association OR relationship)",
        f"{variable1} AND {variable2} AND (risk factor OR predictor)",
        
        # Medium specificity - methodological
        f"{variable1} AND {variable2} AND (longitudinal OR prospective OR cohort)",
        f"{variable1} AND {variable2} AND (correlation OR related)",
        
        # Broader search
        f"{variable1} AND {variable2}",
        f'"{variable1}" AND "{variable2}"',
    ]
    
    return queries


def search_pubmed(query: str, max_results: int = 5, email: Optional[str] = None) -> List[str]:
    """
    Search PubMed and return list of PMIDs.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        email: Email for NCBI identification (optional)
    
    Returns:
        List of PubMed IDs (PMIDs)
    """
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'xml'
    }
    
    if email:
        params['email'] = email
    
    try:
        response = requests.get(ESEARCH_URL, params=params)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        pmids = [id_elem.text for id_elem in root.findall('.//Id')]
        
        return pmids
        
    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def fetch_pubmed_abstracts(pmids: List[str], email: Optional[str] = None) -> List[Dict]:
    """
    Fetch abstracts for given PMIDs from PubMed.
    
    Args:
        pmids: List of PubMed IDs
        email: Email for NCBI identification
    
    Returns:
        List of dictionaries with paper information
    """
    if not pmids:
        return []
    
    params = {
        'db': 'pubmed',
        'id': ','.join(pmids),
        'retmode': 'xml',
        'rettype': 'abstract'
    }
    
    if email:
        params['email'] = email
    
    try:
        response = requests.get(EFETCH_URL, params=params)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        papers = []
        
        for article in root.findall('.//PubmedArticle'):
            paper = _extract_paper_info(article)
            if paper:
                papers.append(paper)
        
        return papers
        
    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return []


def _extract_paper_info(article_elem) -> Optional[Dict]:
    """Extract information from PubmedArticle XML element."""
    try:
        paper = {}
        
        # PMID
        pmid_elem = article_elem.find('.//PMID')
        paper['pmid'] = pmid_elem.text if pmid_elem is not None else 'Unknown'
        
        # Title
        title_elem = article_elem.find('.//ArticleTitle')
        paper['title'] = title_elem.text if title_elem is not None else 'No title'
        
        # Abstract
        abstract_parts = []
        abstract_elem = article_elem.find('.//Abstract')
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall('.//AbstractText'):
                text = text_elem.text or ''
                if text:
                    abstract_parts.append(text)
        
        paper['abstract'] = ' '.join(abstract_parts) if abstract_parts else 'No abstract available'
        
        # Authors
        authors = []
        author_list = article_elem.find('.//AuthorList')
        if author_list is not None:
            for author in author_list.findall('.//Author'):
                last_name = author.find('.//LastName')
                if last_name is not None:
                    authors.append(last_name.text)
        
        paper['authors'] = '; '.join(authors[:3]) if authors else 'Unknown'  # First 3 authors
        
        # Journal
        journal_elem = article_elem.find('.//Journal/Title')
        paper['journal'] = journal_elem.text if journal_elem is not None else 'Unknown'
        
        # Year
        pub_date = article_elem.find('.//PubDate')
        if pub_date is not None:
            year = pub_date.find('.//Year')
            paper['year'] = year.text if year is not None else 'Unknown'
        else:
            paper['year'] = 'Unknown'
        
        return paper
        
    except Exception as e:
        print(f"Error extracting paper: {e}")
        return None


def search_and_fetch_pubmed(variable1: str, variable2: str, 
                            max_papers: int = 10, 
                            email: Optional[str] = None) -> str:
    """
    Complete PubMed search workflow with query rewriting.
    
    Args:
        variable1: First variable name
        variable2: Second variable name
        max_papers: Maximum number of papers to retrieve
        email: Email for NCBI identification
    
    Returns:
        Combined text from all retrieved abstracts
    """
    print(f"🔍 Searching PubMed for: {variable1} ↔ {variable2}")
    
    # Generate query variations
    queries = rewrite_query_for_pubmed(variable1, variable2)
    
    all_papers = []
    seen_pmids = set()
    
    # Try queries in order until we get enough papers
    for query in queries:
        if len(all_papers) >= max_papers:
            break
            
        print(f"   Trying query: {query}")
        pmids = search_pubmed(query, max_results=5, email=email)
        
        if pmids:
            # Filter out duplicates
            new_pmids = [pmid for pmid in pmids if pmid not in seen_pmids]
            
            if new_pmids:
                time.sleep(0.5)  # Be respectful to NCBI servers
                papers = fetch_pubmed_abstracts(new_pmids, email=email)
                
                for paper in papers:
                    if paper['pmid'] not in seen_pmids:
                        seen_pmids.add(paper['pmid'])
                        all_papers.append(paper)
                
                print(f"   ✓ Found {len(papers)} papers")
    
    print(f"📚 Total papers retrieved: {len(all_papers)}")
    
    # Combine all abstracts into single text
    if all_papers:
        pubmed_text = "\n\n".join([
            f"[{paper['authors']} ({paper['year']})] {paper['title']}: {paper['abstract']}"
            for paper in all_papers
        ])
        return pubmed_text
    else:
        return ""

# This method is like split data and create vector store retriever but combines both sources
def create_combined_retriever(causenet_text: str, pubmed_text: str, min_text_length: int = 200) -> Optional[object]:
    """
    Create a unified retriever combining CauseNet and PubMed sources.
    
    Args:
        causenet_text: Text from CauseNet sources
        pubmed_text: Text from PubMed abstracts
        min_text_length: Minimum combined text length to create retriever (default: 200 chars)
    
    Returns:
        Langchain retriever object or None if no text available or too short
    """
    # Combine texts
    combined_text = ""
    
    if causenet_text:
        combined_text += f"=== CauseNet Knowledge ===\n{causenet_text}\n\n"
    
    if pubmed_text:
        combined_text += f"=== Scientific Literature (PubMed) ===\n{pubmed_text}"
    
    # Check if we have enough meaningful content
    if not combined_text or len(combined_text.strip()) < min_text_length:
        print(f"⚠️  Insufficient context (length: {len(combined_text)} chars, min: {min_text_length})")
        print("   → Will use LLM's internal knowledge only")
        return None
    
    print(f"📊 Creating combined retriever (text length: {len(combined_text)} chars)")
    
    # Create document
    document = Document(page_content=combined_text)
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Larger chunks for better context
        chunk_overlap=50
    )
    splits = text_splitter.split_documents([document])
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create in-memory vector store (no persistence to avoid contamination from previous queries)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
        # NO persist_directory - each query gets fresh retriever with only relevant context
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7}  # Retrieve more chunks for better coverage
    )
    
    return retriever
