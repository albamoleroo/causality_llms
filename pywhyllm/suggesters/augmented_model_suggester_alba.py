import logging
import os
import re

from .simple_model_suggester import SimpleModelSuggester
from pywhyllm.utils.data_loader import download_causenet, load_causenet_json, create_causenet_dict
#from pywhyllm.utils.augmented_model_suggester_utils_alba_logprobs import *
from pywhyllm.utils.augmented_model_suggester_utils_alba import *


class AugmentedModelSuggester(SimpleModelSuggester):
    """
    A class that extends SimpleModelSuggester and provides methods for suggesting 
    causal relationships between variables by leveraging both CauseNet dataset 
    and PubMed scientific literature for Retrieval Augmented Generation (RAG).

    Methods:
    - suggest_pairwise_relationship(variable1: str, variable2: str, use_pubmed: bool) -> List[str]:
        Suggests the causal relationship between two variables using combined knowledge
        from CauseNet and optionally PubMed.
    """

    def __init__(self, llm, file_path: str = 'data/causenet-precision.jsonl.bz2', 
                 pubmed_email: str = None, langchain_llm=None):
        """
        Initialize the AugmentedModelSuggester with a language model and download CauseNet data.

        Args:
            llm: The guidance language model instance to be used for querying.
            file_path (str, optional): Path to save the downloaded CauseNet JSONL file.
                                      Defaults to 'data/causenet-precision.jsonl.bz2'.
            pubmed_email (str, optional): Email for PubMed API requests (recommended by NCBI).
            langchain_llm (optional): LangChain-compatible LLM for RAG queries.
                                     If None, will create a default OpenAI instance.
        """

        super().__init__(llm)
        self.file_path = file_path
        self.pubmed_email = pubmed_email
        self.langchain_llm = langchain_llm  # Store LangChain LLM separately

        logging.basicConfig(level=logging.INFO)
        
        # Check if file already exists locally (same behavior as original augmented_model_suggester)
        if os.path.exists(file_path):
            print(f"✓ CauseNet found locally at {file_path}")
            json_data = load_causenet_json(file_path)
            self.causenet_dict = create_causenet_dict(json_data)
        else:
            # File not found - try to download
            url = "https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2"
            print(f"CauseNet not found at {file_path}, attempting download...")
            success = download_causenet(url, file_path)

            if success:
                print(f"✓ CauseNet downloaded to {file_path}")
                json_data = load_causenet_json(file_path)
                self.causenet_dict = create_causenet_dict(json_data)
            else:
                print("✗ CauseNet download failed")
                print(f"   Manual download: {url}")
                print(f"   Save to: {os.path.abspath(file_path)}")
                self.causenet_dict = {}

    def suggest_pairwise_relationship(self, variable1: str, variable2: str, 
                                     use_pubmed: bool = True,
                                     max_pubmed_papers: int = 10,
                                     return_prompt: bool = False,
                                     confidence_level: bool = True):
        """
        Suggests a cause-and-effect relationship between two variables using combined
        knowledge from CauseNet and PubMed scientific literature.
        
        The method:
        1. Searches CauseNet for matching causal pairs
        2. If use_pubmed=True, searches PubMed with query rewriting
        3. Combines both sources into a unified retriever
        4. Queries the LLM with augmented context

        Args:
            variable1 (str): The name of the first variable.
            variable2 (str): The name of the second variable.
            use_pubmed (bool): Whether to include PubMed search (default: True).
            max_pubmed_papers (int): Maximum number of PubMed papers to retrieve.
            return_prompt (bool): If True, returns dict with result and system_prompt (default: False).
            confidence_level (bool): If True, instructs LLM to provide confidence score (default: False).

        Returns:
            If return_prompt=False and confidence_level=False:
                list: A list containing [cause_variable, effect_variable, reasoning_description].
                     If no relationship, first two elements will be None.
            If return_prompt=False and confidence_level=True:
                tuple: (result_list, confidence_score) where result_list is as above and 
                       confidence_score is a float between 0-1 or None if not found.
            If return_prompt=True and confidence_level=False:
                dict: Dictionary with keys:
                    - 'result': The [cause, effect, reasoning] list
                    - 'system_prompt': The complete system prompt sent to LLM
                    - 'causenet_text': Raw CauseNet text
                    - 'pubmed_text': Raw PubMed text
            If return_prompt=True and confidence_level=True:
                dict: Same as above plus:
                    - 'confidence_score': Float between 0-1 or None if not found
        """
        
        print(f"\n{'='*60}")
        print(f"🔬 Analyzing: {variable1} ↔ {variable2}")
        print(f"{'='*60}")
        
        # Step 1: Search CauseNet
        print("\n📖 Step 1: Searching CauseNet...")
        causenet_result = find_top_match_in_causenet(self.causenet_dict, variable1, variable2)
        causenet_text = ""
        
        if causenet_result:
            causenet_text = get_source_text(causenet_result)
            print(f"   ✓ Found CauseNet match (text length: {len(causenet_text)} chars)")
        else:
            print("   ✗ No CauseNet match found")
        
        # Step 2: Search PubMed 
        pubmed_text = ""
        if use_pubmed:
            print("\n📚 Step 2: Searching PubMed...")
            pubmed_text = search_and_fetch_pubmed(
                variable1, 
                variable2, 
                max_papers=max_pubmed_papers,
                email=self.pubmed_email
            )
            
            if pubmed_text:
                print(f"   ✓ Retrieved PubMed literature (text length: {len(pubmed_text)} chars)")
            else:
                print("   ✗ No PubMed papers found")
        else:
            print("\n📚 Step 2: Skipping PubMed search")
        
        # Step 3: Create combined retriever or use default
        print("\n🤖 Step 3: Querying LLM...")
        
        # Prepare kwargs for query_llm based on whether langchain_llm is provided
        llm_kwargs = {
            'llm': self.langchain_llm, 
            'return_prompt': return_prompt,
            'confidence_level': confidence_level
        } if self.langchain_llm else {
            'return_prompt': return_prompt,
            'confidence_level': confidence_level
        }
        
        system_prompt = None  # Will store the system prompt if return_prompt=True
        
        if causenet_text or pubmed_text:
            # Create combined retriever
            retriever = create_combined_retriever(causenet_text, pubmed_text)
            
            if retriever:
                combined_text = (causenet_text + "\n\n" + pubmed_text).strip()
                result_data = query_llm(variable1, variable2, combined_text, retriever, **llm_kwargs)
                
                if return_prompt:
                    response, system_prompt = result_data
                else:
                    response = result_data
                
                print("   ✓ LLM response generated with RAG augmentation")
            else:
                result_data = query_llm(variable1, variable2, **llm_kwargs)
                
                if return_prompt:
                    response, system_prompt = result_data
                else:
                    response = result_data
                
                print("   ⚠ LLM response generated without augmentation")
        else:
            # No external knowledge found, use LLM's internal knowledge
            result_data = query_llm(variable1, variable2, **llm_kwargs)
            
            if return_prompt:
                response, system_prompt = result_data
            else:
                response = result_data
            
            print("   ℹ LLM response generated using internal knowledge only")
        
        # Step 4: Parse response
        print("\n📋 Step 4: Parsing result...")
        
        # Convert AIMessage to string if needed
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        answer = re.findall(r'<answer>(.*?)</answer>', response_text)
        answer = [ans.strip() for ans in answer]
        answer_str = "".join(answer)
        
        # Extract confidence and strength scores if confidence_level was requested
        confidence_score = None
        strength_score = None
        if confidence_level:
            # Extract confidence
            confidence_match = re.findall(r'<confidence>(.*?)</confidence>', response_text)
            if confidence_match:
                try:
                    confidence_score = float(confidence_match[0])
                    confidence_score = max(0.0, min(1.0, confidence_score))  # Clamp to [0, 1]
                    print(f"   ℹ Confidence score: {confidence_score:.3f}")
                except ValueError:
                    print(f"   ⚠ Could not parse confidence score: {confidence_match[0]}")
                    confidence_score = None
            
            # Extract strength
            strength_match = re.findall(r'<strength>(.*?)</strength>', response_text)
            if strength_match:
                try:
                    strength_score = float(strength_match[0])
                    strength_score = max(0.0, min(1.0, strength_score))  # Clamp to [0, 1]
                    print(f"   ℹ Strength score: {strength_score:.3f}")
                except ValueError:
                    print(f"   ⚠ Could not parse strength score: {strength_match[0]}")
                    strength_score = None

        if answer_str == "A":
            result = [variable1, variable2, response_text]
            print(f"   → Causal direction: {variable1} → {variable2}")
        elif answer_str == "B":
            result = [variable2, variable1, response_text]
            print(f"   → Causal direction: {variable2} → {variable1}")
        elif answer_str == "C":
            result = [None, None, response_text]
            print(f"   → No causal relationship detected")
        else:
            print(f"   ✗ Invalid LLM response: {answer_str}")
            raise ValueError(f"Invalid answer from LLM: {answer_str}")
        
        print(f"{'='*60}\n")
        
        # Return result with or without system prompt
        if return_prompt:
            result_dict = {
                'result': result,
                'system_prompt': system_prompt,
                'causenet_text': causenet_text,
                'pubmed_text': pubmed_text
            }
            if confidence_level:
                result_dict['confidence_score'] = confidence_score
            return result_dict
        else:
            if confidence_level:
                return result, confidence_score
            return result


    def suggest_pairwise_relationship_logprobs(self, variable1: str, variable2: str,
                                            use_pubmed: bool = True,
                                            max_pubmed_papers: int = 10,
                                            openai_client=None,
                                            model_name="gpt-4o-mini",
                                            temperature=0.3,
                                            log_probs=True,
                                            confidence_level=True):
        """
        Suggests a cause-and-effect relationship between two variables using combined
        knowledge from CauseNet and PubMed, with support for logprobs extraction.
        
        This method combines external knowledge sources (CauseNet + PubMed) with
        direct OpenAI API calls to extract detailed log probabilities for uncertainty
        quantification.
        
        Args:
            variable1 (str): The name of the first variable.
            variable2 (str): The name of the second variable.
            use_pubmed (bool): Whether to include PubMed search (default: True).
            max_pubmed_papers (int): Maximum number of PubMed papers to retrieve.
            openai_client: OpenAI client for direct API access (required for logprobs).
            model_name (str): Model name to use with OpenAI client (default: "gpt-4o-mini").
            temperature (float): Temperature for generation (default: 0.3).
            log_probs (bool): Whether to extract log probabilities (default: True).
            confidence_level (bool): Whether to request confidence and strength scores (default: True).

        Returns:
            dict: Dictionary containing:
                - 'result': List [cause_variable, effect_variable, reasoning_description]
                - 'response': Full LLM response text
                - 'answer': Letter answer ("A", "B", or "C")
                - 'confidence_score': Float 0-1 (if confidence_level=True)
                - 'strength_score': Float 0-1 (if confidence_level=True)
                - 'causenet_text': Raw CauseNet context text
                - 'pubmed_text': Raw PubMed context text
                - 'logprobs': Raw logprobs data (if log_probs=True)
                - 'answer_token_logprob': Logprob of answer token (if log_probs=True)
                - 'answer_choice_logprobs': Dict of A/B/C logprobs (if log_probs=True)
                - 'answer_tokens_logprobs': List of all answer tokens with logprobs (if log_probs=True)
        """
        
        if log_probs and openai_client is None:
            raise ValueError("openai_client parameter is required when log_probs=True")
        
        print(f"\n{'='*60}")
        print(f"🔬 Analyzing: {variable1} ↔ {variable2}")
        print(f"{'='*60}")
        
        # Step 1: Search CauseNet
        print("\n📖 Step 1: Searching CauseNet...")
        causenet_result = find_top_match_in_causenet(self.causenet_dict, variable1, variable2)
        causenet_text = ""
        
        if causenet_result:
            causenet_text = get_source_text(causenet_result)
            print(f"   ✓ Found CauseNet match (text length: {len(causenet_text)} chars)")
        else:
            print("   ✗ No CauseNet match found")
        
        # Step 2: Search PubMed 
        pubmed_text = ""
        if use_pubmed:
            print("\n📚 Step 2: Searching PubMed...")
            pubmed_text = search_and_fetch_pubmed(
                variable1, 
                variable2, 
                max_papers=max_pubmed_papers,
                email=self.pubmed_email
            )
            
            if pubmed_text:
                print(f"   ✓ Retrieved PubMed literature (text length: {len(pubmed_text)} chars)")
            else:
                print("   ✗ No PubMed papers found")
        else:
            print("\n📚 Step 2: Skipping PubMed search")
        
        # Step 3: Query LLM with logprobs
        print("\n🤖 Step 3: Querying LLM with logprobs...")
        
        if causenet_text or pubmed_text:
            # Create combined retriever
            retriever = create_combined_retriever(causenet_text, pubmed_text)
            
            if retriever:
                combined_text = (causenet_text + "\n\n" + pubmed_text).strip()
                result_dict = query_llm(
                    variable1, 
                    variable2, 
                    source_text=combined_text,
                    retriever=retriever,
                    openai_client=openai_client,
                    model_name=model_name,
                    temperature=temperature,
                    log_probs=log_probs,
                    confidence_level=confidence_level
                )
                print("   ✓ LLM response generated with RAG augmentation")
            else:
                result_dict = query_llm(
                    variable1,
                    variable2,
                    openai_client=openai_client,
                    model_name=model_name,
                    temperature=temperature,
                    log_probs=log_probs,
                    confidence_level=confidence_level
                )
                print("   ⚠ LLM response generated without augmentation")
        else:
            # No external knowledge found, use LLM's internal knowledge
            result_dict = query_llm(
                variable1,
                variable2,
                openai_client=openai_client,
                model_name=model_name,
                temperature=temperature,
                log_probs=log_probs,
                confidence_level=confidence_level
            )
            print("   ℹ LLM response generated using internal knowledge only")
        
        # Step 4: Parse response and build final result
        print("\n📋 Step 4: Parsing result...")
        
        response_text = result_dict['response']
        answer_str = result_dict['answer']
        
        # Display extracted scores
        if confidence_level:
            if result_dict.get('confidence_score') is not None:
                print(f"   ℹ Confidence score: {result_dict['confidence_score']:.3f}")
            if result_dict.get('strength_score') is not None:
                print(f"   ℹ Strength score: {result_dict['strength_score']:.3f}")
        
        if log_probs:
            if result_dict.get('answer_token_logprob') is not None:
                print(f"   ℹ Answer token logprob: {result_dict['answer_token_logprob']:.6f}")
            if result_dict.get('answer_choice_logprobs'):
                print(f"   ℹ Choice logprobs: {result_dict['answer_choice_logprobs']}")

        # Determine causal direction
        if answer_str == "A":
            result = [variable1, variable2, response_text]
            print(f"   → Causal direction: {variable1} → {variable2}")
        elif answer_str == "B":
            result = [variable2, variable1, response_text]
            print(f"   → Causal direction: {variable2} → {variable1}")
        elif answer_str == "C":
            result = [None, None, response_text]
            print(f"   → No causal relationship detected")
        else:
            print(f"   ✗ Invalid LLM response: {answer_str}")
            raise ValueError(f"Invalid answer from LLM: {answer_str}")
        
        print(f"{'='*60}\n")
        
        # Build comprehensive return dictionary
        return_dict = {
            'result': result,
            'response': response_text,
            'answer': answer_str,
            'causenet_text': causenet_text,
            'pubmed_text': pubmed_text
        }
        
        # Add confidence and strength scores if requested
        if confidence_level:
            return_dict['confidence_score'] = result_dict.get('confidence_score')
            return_dict['strength_score'] = result_dict.get('strength_score')
        
        # Add logprobs data if requested
        if log_probs:
            return_dict.update({
                'logprobs': result_dict.get('logprobs'),
                'answer_token_logprob': result_dict.get('answer_token_logprob'),
                'answer_choice_logprobs': result_dict.get('answer_choice_logprobs'),
                'answer_tokens_logprobs': result_dict.get('answer_tokens_logprobs')
            })
        
        return return_dict