from typing import List
import guidance
import re
import itertools
from guidance import system, user, assistant, gen
from inspect import cleandoc

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
        # model_name="GPT-4o-2024-05-13"
    def __init__(self, llm=None, model_name="gpt-4o-mini"):
        if llm is None:
            self.llm = guidance.models.OpenAI(model_name)
        elif isinstance(llm, guidance.models.Model):
            self.llm = llm
        else:
            raise ValueError("llm must be either a guidance model instance or None.")

    # new ver
    def suggest_pairwise_relationship(self, variable1: str, variable2: str):
        """
            Suggests a cause-and-effect relationship between two variables.

            Args:
                variable1 (str): The name of the first variable.
                variable2 (str): The name of the second variable.

            Returns:
                list: A list containing the suggested cause variable, the suggested effect variable, and a description of the reasoning behind the suggestion.  If there is no relationship between the two variables, the first two elements will be None.
            """

        lm = self.llm
        with system():
            lm += "You are a helpful assistant for causal reasoning."

        with user():
            prompt_str = f"""Which cause-and-effect-relationship is more likely? Provide reasoning and give your final answer (A, B, or C) in <answer> </answer> tags with the letter only and no whitespaces.
            A. {variable1} causes {variable2} B. {variable2} causes {variable1} C. neither {variable1} nor {variable2} cause each other."""
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
            return [None, None, description]  # maybe we want to save the description in this case too
        else:
            assert False, "Invalid answer from LLM: " + answer_str

    def suggest_relationships(self, variables: List[str]):
        """
        Given a list of variables, suggests relationships between them by querying for pairwise relationships.

        Args:
            variables (List[str]): A list of variable names.

        Returns:
            dict: A dictionary of edges found between variables, where the keys are tuples representing the causal relationship between two variables,
            and the values are the strength of the relationship.
        """
        relationships = {}
        total = (len(variables) * (len(variables) - 1) / 2)
        i = 0
        for (var1, var2) in itertools.combinations(variables, 2):
            i += 1
            print(f"{i}/{total}: Querying for relationship between {var1} and {var2}")
            y = list(self.suggest_pairwise_relationship(var1, var2))
            if (y[0] == None):
                print(f"\tNo relationship found between {var1} and {var2}")
                continue
            print(f"\t{y[0]} causes {y[1]}")
            relationships[(y[0], y[1])] = y[2]

        return relationships

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