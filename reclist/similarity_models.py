from abc import ABC, abstractmethod
import json
import os

class SimilarityModel(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def similarity_binary(self, query, target, *args, **kwargs):
        pass

    @abstractmethod
    def similarity_gradient(self, query, target, *args, **kwargs):
        pass


class FakeSimilarityModel(SimilarityModel):

    """
    TODO: this is a stub, to showcase the idea of a similarity model
    and how it can be implemented in different ways
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_threshold = kwargs.get("similarity_threshold", 0.5)
        return

    def similarity_binary(self, query, target, *args, **kwargs) -> bool:
        """
        We can binarize the similarity by taking the threshold over
        the distance in the latent space
        """
        similarity_score = self.similarity_gradient(query, target, *args, **kwargs)
        return similarity_score > self.similarity_threshold

    def similarity_gradient(self, query, target, *args, **kwargs) -> float:
        """
        We return the distance in the latent space
        """
        # TODO: replace this with the actual distance
        from random import random
        return random()


class GPT3SimilarityModel(SimilarityModel):

    API_URL = 'https://api.openai.com/v1/completions'
    MODEL = "text-davinci-002"
    TEMPERATURE = 0
    SIMILARITY_PROMPT = '''
        Imagine you are shopping for fashion products in a fashion store.
        Your best friend tells you to buy something as close as possible to this item:

        {}.

        The shopping assistant proposes the following alternative:

        {}.

        Is this second product similar enough to the one suggested by your friend? Provide a yes/no answer.
    '''

    """
    TODO: this is a stub, to showcase the idea of a similarity model
    and how it can be implemented in different ways
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # get the key from kwargs or fall back on envs
        self.api_key = kwargs.get("api_key", os.environ["OPENAI_API_KEY"])
        # you can override the model and temperature
        self.MODEL = kwargs.get("model", self.MODEL)
        self.TEMPERATURE = kwargs.get("temperature", self.TEMPERATURE)
        self.SIMILARITY_PROMPT = kwargs.get("similarity_prompt", self.SIMILARITY_PROMPT)
        return

    def similarity_binary(self, query, target, *args, **kwargs) -> bool:
        import requests
        # serialize the query and target
        query_str = ''
        for k, v in query.items():
            query_str += '{}: {} | '.format(k.strip(), v.replace('\n', ' ').strip())
        target_str = ''
        for k, v in target.items():
            target_str += '{}: {} | '.format(k.strip(), v.replace('\n', ' ').strip())
        final_prompt = self.SIMILARITY_PROMPT.format(query_str, target_str).strip()
        data =  {
                "model": self.MODEL,
                "prompt": final_prompt,
                "temperature": self.TEMPERATURE,
                "max_tokens": 10
                }
        headers = {
            'content-type': 'application/json',
            "Authorization": "Bearer {}".format(self.api_key)
            }
        r = requests.post(self.API_URL, data=json.dumps(data), headers=headers)
        # NOTE: we break if API call fails, alternative behavior are possible of course
        if r.status_code != 200:
            raise Exception("Error in GPT3 API call: {}".format(r.text))

        completion = json.loads(r.text)['choices'][0]['text']
        if kwargs.get("verbose", False):
            print("Query: {}".format(query_str))
            print("Target: {}".format(target_str))
            print("Prompt: {}".format(final_prompt))
            print("Completion: {}".format(completion))
        # if the first word is Yes, we return True
        first_word = completion.strip().split(" ")[0].lower()

        return first_word == "yes"

    def similarity_gradient(self, query, target, *args, **kwargs) -> float:
        raise Exception("No gradient is available for GPT3")

# test the similarity model with open ai :clownface:
#sim_model = GPT3SimilarityModel(api_key=os.environ["OPENAI_API_KEY"])
#p1 = {
#    "name": "logo-print cotton cap",
#    "brand": 'Palm Angels',
#    "description": '''
#    Known for a laid-back aesthetic, Palm Angels knows how to portray its Californian inspiration. This classic cap carries the brand's logo printed on the front, adding a touch of recognition to a relaxed look.
#    '''
#}
#p2 = {
#    "name": "monogram badge cap",
#    "brand": 'Balmain',
#    "description": '''
#    Blue cotton monogram badge cap from Balmain featuring logo patch to the front, mesh detailing, fabric-covered button at the crown and adjustable fit.
#    '''
#}
#similarity_judgement = sim_model.similarity_binary(p1, p2, verbose=False)
#print("P1 {} and P2 {} are similar: {}".format(p1["name"], p2["name"], similarity_judgement))



