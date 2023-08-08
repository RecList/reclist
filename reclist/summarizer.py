import guidance
from typing import Optional

guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")
PROMPT = """
{{#system~}}
Assume you are a Data Scientist assistant helping Data Science practicioners evaluate their recommender system models.
You will be given a list of metrics and you should do 2 tasks:
1. Help summarize the finding
2. Provide advice on what to do that could increase the metrics
You will report your finding being specific for example referring the actual metric and values while being succinct using bullets points.
For example you can look at correlations between metrics, outliers or range of the metrics to draw conclusion.
As a Data Scientist you do not need to report on all the metrics but only on the one providing incremental value to the analysis.
Therefore, it is key to only output information that provide value maximizing the value while minimizing the verbosity of it.
Do not hesitate to group multiple metrics into one bullet point if they are going towards the same conclusion.
It will only make your reasoning stronger.
You should aim for each bullet point to do no more than one sentence so digesting this information is easy and fast.
Finally, you do not need to explain what the metrics are as you are already speaking to an expert.
Do not hesitate to use technical jargon if it helps you to be more concise.
The metrics follow an array of json with each element having theses keys:
    1. "name" This is the name of the metric it follows this pattern <metric_name>_<slice_name> where slice name is optional.
    2. "description" is an optional description entered by the user
    3. "result" this is where you will get the metric value or additional slice from the metric "name"
    4. "display_type" Ignore this

In addition here is a mapping of the metrics name:
MRR means mean reciprocal rank
HIT_RATE means hit rate
MRED means miss rate equality difference
BEING_LESS_WRONG compute the cosine similarity between the true label and the predictions.
MR means miss rate which is the opposite of HIT_RATE
{{#if compare_statistics}}
You will be given 2 sets of model metrics to compare 2 different models.
Please focus on the comparison so the Data Scientist can draw conclusion.
{{/if}}
{{~/system}}
{{#user~}}
Given that I have a model that I named {{model_name}} and this statistics:
 {{statistics}}
{{#if compare_statistics}}
In addition, my second model is named {{compare_model_name}} and has this statistics:
{{compare_statistics}}
{{/if}}
Please summarize your findings.
{{~/user}}

{{#assistant~}}
{{gen 'out' temperature=0}}
{{~/assistant}}
"""
PROGRAM = guidance(PROMPT)


def summarize_statistics(
    model_name: str,
    statistics: list,
    compare_model_name: Optional[str] = None,
    compare_statistics: Optional[list] = None,
) -> Optional[str]:
    """This function use OpenAI to summarize or compare 2 models statistics from reclist

    `compare_model_name` and `compare_statistics` are optional and only used if you want to compare 2 models.
    If not used it will provide summary on one model defined by `model_name` `statistics`


    Args:
        model_name (str): Model Name
        statistics (list): List of statistics as defined by reclist
        compare_model_name (Optional[str]): Optional, Model Name to compare
        compare_statistics (Optional[list]): Optional, statistics as defined in reclist to compare

    Returns:
        String that summarize the model statistics or comparison between model and target model

    Raises:
        ValueError: If one of the two is not None while the other is `compare_model_name` or `compare_statistics`
    """
    if compare_model_name is not None and compare_statistics is None:
        raise ValueError(
            "You have specified a compare_model_name without compare_statistics"
        )
    if compare_model_name is None and compare_statistics is not None:
        raise ValueError(
            "You have specified compare_statistics without a compare_model_name"
        )
    summary = PROGRAM(**locals())
    return summary["out"]

