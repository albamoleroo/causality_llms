from typing import List
import guidance
import re
import itertools
import math
from guidance import system, user, assistant, gen
from inspect import cleandoc
from openai import OpenAI

class SimpleModelSuggester:
    """
    A class that provides methods for suggesting causal relationships and confounding factors between variables.

    This class uses the guidance library to interact with LLMs, and assumes that guidance.llm has already been initialized to the user's preferred LLM

    Methods:
    - suggest_pairwise_relationship(variable1: str, variable2: str) -> List[str]:
        Suggests the causal relationship between two variables and returns a list containing the cause, effect, and a description of the relationship.
    - suggest_relationships(variables: List[str]) -> Dict[Tuple[str, str], str]:
        Suggests the causal relationships between all pairs of variables in a list and returns a dictionary containing the cause-effect pairs and their descriptions.
    - suggest_confounders(variables: List[str], treatment: str, outcome: str) -> List[str]:
        Suggests the confounding factors that might influence the relationship between a treatment and an outcome, given a list of variables that have already been considered.
    """

    def __init__(self, llm=None, model_name="gpt-4o-mini"):
        if llm is None:
            self.llm = guidance.models.OpenAI(model_name)
        elif isinstance(llm, guidance.models.Model):
            self.llm = llm
        else:
            raise ValueError("llm must be either a guidance model instance or None.")

    # #original del repo funciona bien con mis tests
    # def suggest_pairwise_relationship(self, variable1: str, variable2: str):
    #     """
    #         Suggests a cause-and-effect relationship between two variables.

    #         Args:
    #             variable1 (str): The name of the first variable.
    #             variable2 (str): The name of the second variable.

    #         Returns:
    #             list: A list containing the suggested cause variable, the suggested effect variable, and a description of the reasoning behind the suggestion.  If there is no relationship between the two variables, the first two elements will be None.
    #         """

    #     lm = self.llm
    #     with system():
    #         lm += "You are a helpful assistant for causal reasoning."

    #     with user():
    #         prompt_str = f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.
    #         A. {variable1} causes {variable2} B. {variable2} causes {variable1} C. neither {variable1} nor {variable2} cause each other."""
    #         lm += cleandoc(prompt_str)

    #     with assistant():
    #         lm += gen("description")

    #     description = lm['description']
    #     answer = re.findall(r'<answer>(.*?)</answer>', description)
    #     answer = [ans.strip() for ans in answer]
    #     answer_str = "".join(answer)

    #     if answer_str == "A":
    #         return [variable1, variable2, description]
    #     elif answer_str == "B":
    #         return [variable2, variable1, description]
    #     elif answer_str == "C":
    #         return [None, None, description]  # maybe we want to save the description in this case too
    #     else:
    #         assert False, "Invalid answer from LLM: " + answer_str


    def suggest_pairwise_relationship(self, variable1: str, variable2: str, all_variables: List[str] = None):
        """
        Suggests a cause-and-effect relationship between two variables.
        Considers whether the relationship might be mediated by other variables.

        Args:
            variable1 (str): The name of the first variable.
            variable2 (str): The name of the second variable.
            all_variables (List[str], optional): List of all variables in the analysis.
                If provided, the LLM will check for mediating relationships.

        Returns:
            list: A list containing the suggested cause variable, the suggested effect variable, 
            and a description of the reasoning behind the suggestion. If there is no DIRECT 
            relationship between the two variables, the first two elements will be None.
        """

        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            # Build mediation check instruction if all_variables provided
            if all_variables is not None:
                other_vars = [v for v in all_variables if v not in [variable1, variable2]]
                mediator_instruction = f"""variables that could act as mediators in this edge: {other_vars},

                Example:
                Variables: Exercise, Weight Loss
                Potential mediators: Calorie Intake
                Reasoning: Exercise can influence calorie intake, which in turn affects weight loss. Since the effect of exercise on weight loss may be explained through changes in calorie intake, the relationship is mediated by another variable.
                Final answer: <answer>C</answer>
                """
            else:
                mediator_instruction = ""

            prompt_str = f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.
            A. {variable1} causes {variable2} B. {variable2} causes {variable1} C. neither {variable1} nor {variable2} cause each other or the relationship is mediated by another variable.{mediator_instruction}"""            
            lm += cleandoc(prompt_str)

        with assistant():
            lm += gen("description")

        description = lm['description']
        answer = re.findall(r'<answer>(.*?)</answer>', description)
        answer = [ans.strip() for ans in answer]
        answer_str = "".join(answer)

        if answer_str == "A":
            return [variable1, variable2, description]
        elif answer_str == "B":
            return [variable2, variable1, description]
        elif answer_str == "C":
            return [None, None, description]
        else:
            assert False, "Invalid answer from LLM: " + answer_str

    def suggest_relationships(self, variables: List[str]):
        """
        Given a list of variables, suggests relationships between them by querying for pairwise relationships.
        Checks for mediated relationships to avoid creating spurious direct edges.

        Args:
            variables (List[str]): A list of variable names.

        Returns:
            dict: A dictionary of DIRECT edges found between variables, where the keys are tuples 
            representing the causal relationship between two variables, and the values are the 
            descriptions of the relationship.
        """
        relationships = {}
        total = (len(variables) * (len(variables) - 1) / 2)
        i = 0
        for (var1, var2) in itertools.combinations(variables, 2):
            i += 1
            print(f"{i}/{total}: Querying for relationship between {var1} and {var2}")
            # Pass all variables to enable mediation checking
            y = list(self.suggest_pairwise_relationship(var1, var2, all_variables=variables))
            if (y[0] == None):
                print(f"\tNo direct relationship found between {var1} and {var2}")
                continue
            print(f"\t{y[0]} directly causes {y[1]}")
            relationships[(y[0], y[1])] = y[2]

        return relationships

    # Alba suggest pairwise relationships with logprobs and confidence level booleans
    def suggest_pairwise_relationship_with_logprobs_flexible(self, variable1: str, variable2: str, openai_client=None, model_name="gpt-4o-mini", log_probs=True, confidence_level=False):
        """
        Suggests a cause-and-effect relationship with optional detailed log probabilities.
        Uses OpenAI client directly for full logprob access when enabled.
        
        Args:
            variable1 (str): The name of the first variable.
            variable2 (str): The name of the second variable.
            openai_client: Optional OpenAI client instance. If None, uses guidance model.
            model_name (str): Model name to use with OpenAI client.
            log_probs (bool): Whether to calculate log probabilities. Default True.
            confidence_level (bool): Whether to request confidence score. Default False.
            
        Returns:
            dict: Contains 'result', 'description', 'answer', optionally 'logprobs' data, and optionally 'confidence'.
        """
        # If no OpenAI client provided, try to create one from environment
        if openai_client is None:
            import os
            # Try to extract connection details from guidance model
            if hasattr(self.llm, 'engine'):
                try:
                    openai_client = OpenAI(
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_BASE_URL")
                    )
                except:
                    raise ValueError("Could not create OpenAI client. Please pass openai_client parameter.")
        
        # Confidence instruction
        confidence_instruction = ""
        if confidence_level:
            confidence_instruction = " Also provide a confidence score between 0 and 1 within <confidence></confidence> tags, regardless of whether the answer is A, B, or C."
        
        # Construct the prompt
        prompt = f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.{confidence_instruction}
    A. {variable1} causes {variable2} 
    B. {variable2} causes {variable1} 
    C. neither {variable1} nor {variable2} cause each other."""
        
        # Make API call with conditional logprobs
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for causal reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            logprobs=log_probs,  # Use the boolean parameter
            top_logprobs=3 if log_probs else None,  # Only request top logprobs if needed
            max_tokens=200
        )
        
        choice = response.choices[0]
        description = choice.message.content
        
        # Extract answer
        answer = re.findall(r"<answer>(.*?)</answer>", description)
        answer = [ans.strip() for ans in answer]
        answer_str = "".join(answer) if answer else ""

        # Extract confidence if requested
        confidence_score = None
        if confidence_level:
            confidence_match = re.findall(r"<confidence>(.*?)</confidence>", description)
            if confidence_match:
                try:
                    confidence_score = float(confidence_match[0].strip())
                except ValueError:
                    confidence_score = None

        # Determine result based on the chosen answer
        if answer_str == "A":
            result = [variable1, variable2, description]
        elif answer_str == "B":
            result = [variable2, variable1, description]
        elif answer_str == "C":
            result = [None, None, description]
        else:
            result = [None, None, description]

        # Base return dictionary
        return_dict = {
            'result': result,
            'description': description,
            'answer': answer_str,
        }
        
        # Add confidence score if requested
        if confidence_level:
            return_dict['confidence'] = confidence_score
        
        # Only process logprobs if requested
        if log_probs:
            logprobs_data = []
            if getattr(choice, "logprobs", None) and getattr(choice.logprobs, "content", None):
                logprobs_data = choice.logprobs.content

            answer_token_logprob = None
            answer_choice_logprobs = {}
            answer_tokens_logprobs = []

            if logprobs_data:
                reconstructed_text = ""
                token_spans = []
                for token_item in logprobs_data:
                    token_text = getattr(token_item, "token", "") or ""
                    start_idx = len(reconstructed_text)
                    reconstructed_text += token_text
                    end_idx = len(reconstructed_text)
                    token_spans.append((start_idx, end_idx, token_item))

                match = re.search(r"<answer>(.*?)</answer>", reconstructed_text, flags=re.DOTALL)
                if match:
                    content_start, content_end = match.span(1)
                    for start_idx, end_idx, token_item in token_spans:
                        overlaps_answer = start_idx < content_end and end_idx > content_start
                        if overlaps_answer:
                            token_text = getattr(token_item, "token", "") or ""
                            token_logprob = getattr(token_item, "logprob", None)
                            answer_tokens_logprobs.append({
                                "token": token_text,
                                "logprob": token_logprob
                            })

                            token_letter = token_text.strip()
                            if token_letter in {"A", "B", "C"} and token_logprob is not None:
                                answer_choice_logprobs[token_letter] = token_logprob

                            top_alternatives = getattr(token_item, "top_logprobs", None) or []
                            for alt in top_alternatives:
                                alt_token = (getattr(alt, "token", "") or "").strip()
                                if alt_token in {"A", "B", "C"}:
                                    alt_logprob = getattr(alt, "logprob", None)
                                    if alt_logprob is not None:
                                        answer_choice_logprobs[alt_token] = alt_logprob
                        if start_idx <= content_start and end_idx > content_start:
                            if answer_token_logprob is None:
                                answer_token_logprob = getattr(token_item, "logprob", None)

            # Add logprobs data to return dictionary
            return_dict.update({
                'logprobs': logprobs_data,
                'answer_token_logprob': answer_token_logprob,
                'answer_choice_logprobs': answer_choice_logprobs,
                'answer_tokens_logprobs': answer_tokens_logprobs,
            })

        return return_dict

    # This is working propperly but gives you ALL log probs for all tokens (not calculating the answer token)
    # def suggest_pairwise_relationship_with_logprobs(self, variable1: str, variable2: str, openai_client=None, model_name="gpt-4o-mini"):
    #     """
    #     Suggests a cause-and-effect relationship with detailed log probabilities.
    #     Uses OpenAI client directly for full logprob access.
        
    #     Args:
    #         variable1 (str): The name of the first variable.
    #         variable2 (str): The name of the second variable.
    #         openai_client: Optional OpenAI client instance. If None, uses guidance model.
    #         model_name (str): Model name to use with OpenAI client.
            
    #     Returns:
    #         dict: Contains 'result', 'description', 'answer', 'logprobs', and 'confidence_scores'
    #     """
    #     from openai import OpenAI
        
    #     # If no OpenAI client provided, try to create one from environment
    #     if openai_client is None:
    #         import os
    #         # Try to extract connection details from guidance model
    #         if hasattr(self.llm, 'engine'):
    #             try:
    #                 openai_client = OpenAI(
    #                     api_key=os.environ.get("OPENAI_API_KEY"),
    #                     base_url=os.environ.get("OPENAI_BASE_URL")
    #                 )
    #             except:
    #                 raise ValueError("Could not create OpenAI client. Please pass openai_client parameter.")
        
    #     # Construct the prompt
    #     prompt = f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.
    #     A. {variable1} causes {variable2} B. {variable2} causes {variable1} C. neither {variable1} nor {variable2} cause each other."""
        
    #     # Make API call with logprobs
    #     response = openai_client.chat.completions.create(
    #         model=model_name,
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant for causal reasoning."},
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.0,
    #         logprobs=True,
    #         top_logprobs=3,
    #         max_tokens=200
    #     )
        
    #     description = response.choices[0].message.content
    #     logprobs_data = response.choices[0].logprobs.content

    #     # Extract answer
    #     answer = re.findall(r"<answer>(.*?)</answer>", description)
    #     answer = [ans.strip() for ans in answer]
    #     answer_str = "".join(answer) if answer else ""

    #     # Determine result based on the chosen answer
    #     if answer_str == "A":
    #         result = [variable1, variable2, description]
    #     elif answer_str == "B":
    #         result = [variable2, variable1, description]
    #     elif answer_str == "C":
    #         result = [None, None, description]
    #     else:
    #         result = [None, None, description]

    #     return {
    #         'result': result,
    #         'description': description,
    #         'answer': answer_str,
    #         'logprobs': logprobs_data,
    #     }
    
    # new ver
    #  previously: We have already considered the following factors {variables}.  Please do not repeat them.
    def suggest_confounders(self, variables: List[str], exposure: str, outcome: str) -> List[str]:

        """
            Suggests potential confounding factors that might influence the relationship between the main exposure and outcome variables.

            Args:
                variables (List[str]): A list of variables that have already been considered.
                exposure (str): The name of the exposure/treatment variable.
                outcome (str): The name of the outcome variable.

            Returns:
                List[str]: A list of potential confounding factors.
            """

        lm = self.llm

        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            prompt_str = f"""What latent confounding factors might influence the relationship between {exposure} and {outcome}?

            From the available variables list {variables}, list the confounding factors between {exposure} and {outcome} enclosing the name of each factor in <conf> </conf> tags.
            """
            lm += cleandoc(prompt_str)
        with assistant():
            lm += gen("latents")

        latents = lm['latents']
        latents_list = re.findall(r'<conf>(.*?)</conf>', latents)

        return latents_list
    

    
    
    def suggest_confounders_custom(self, variables: List[str], exposure: str, outcome: str) -> List[str]:

        """
            Identifies potential confounding factors from a given list of variables that might influence the relationship between the exposure and outcome variables.
            Custom method created including the confounder definition in the prompt.

            Args:
                variables (List[str]): A list of available variables to consider as potential confounders.
                exposure (str): The name of the exposure/treatment variable.
                outcome (str): The name of the outcome variable.

            Returns:
                List[str]: A list of variables from the input list that are identified as potential confounding factors.
            """

        lm = self.llm

        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            prompt_str = f"""Which variables from the following list might be confounding factors that influence the relationship between {exposure} and {outcome}?

            Available variables: {variables}

            A Confounding for the effect of exposure A on outcome Y is present when the association between A and Y is not entirely due to the causal effect of A on Y
            Any study variable that fulfills these 3 criteria should be evaluated as a possible confounder.
            
            1) A confounder must be an extraneous risk factor for the disease outcome, a different factor from the main exposure under study.
            2) A confounder must be associated with the exposure in the source population of the study participants. This association can be a direct effect of the confounder on the exposure or through the relation of a common cause variable that precedes both the exposure and the confounder.
            3) A confounder must not be affected by the exposure or the disease outcome. In particular, a confounder cannot be an intermediate variable in the causal path between the exposure and the outcome.

            From the available variables list {variables}, list the confounding factors between {exposure} and {outcome} enclosing the name of each factor in <conf> </conf> tags.
            """
            lm += cleandoc(prompt_str)
        with assistant():
            lm += gen("latents")

        latents = lm['latents']
        latents_list = re.findall(r'<conf>(.*?)</conf>', latents)

        return latents_list
    

    def suggest_colliders_custom(self, factors, treatment, outcome):
        """
        Suggests factors that might be colliders between treatment and outcome.
        
        A collider is a variable that is caused by both the treatment and outcome.
        Conditioning on colliders can introduce bias (collider bias or selection bias).
        
        Args:
            factors (list): List of available factors to consider
            treatment (str): Treatment/exposure variable
            outcome (str): Outcome variable
            
        Returns:
            list: Factors identified as potential colliders
        """
        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            prompt_str = f"""Which factors in {factors} might be colliders with respect to {treatment} and {outcome}?


            CRITERIA for a variable to be a collider:
            1. {treatment} → variable (treatment causes the variable)
            2. {outcome} → variable (outcome causes the variable)  
            3. BOTH causal arrows point INTO the variable (not away from it)

            EXAMPLE of a TRUE collider:
            - If treatment="smoking" and outcome="alcohol consumption"
            - Possible collider: "liver disease" (because smoking → liver disease AND alcohol → liver disease)

            Think step by step for each factor in {factors}:
            1. Does {treatment} cause this factor?
            2. Does {outcome} cause this factor?
            3. If BOTH answers are YES, then it's a collider
            4. If either answer is NO, then it's NOT a collider

            From the available factors {factors}, identify which ones are TRUE colliders and list them enclosing the name of each factor in <collider> </collider> tags.
            
            If no true colliders exist among the factors, return an empty response.
            """
            lm += cleandoc(prompt_str)
        with assistant():
            lm += gen("colliders")

        colliders = lm['colliders']
        colliders_list = re.findall(r'<collider>(.*?)</collider>', colliders)

        return colliders_list