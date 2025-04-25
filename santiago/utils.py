import re
import pandas as pd
from rouge_score import rouge_scorer, scoring

def get_test_set():
    df = pd.read_csv("data.csv")

    test_df = df.sample(frac=.1)
    test_df.to_csv("test_set.csv", index=False)

    df = df[~df.index.isin(test_df.index)]
    df.to_csv("data.csv", index=False)

CLEAN_VERSION_TEXT_R = re.compile("<.*?>")
VISUAL = re.compile(r'line-(starts|ends)-x="\d+"|line-y="\d+"|number="\d+"|page="\d+"')

def clean_xml(t):
    return re.sub(CLEAN_VERSION_TEXT_R, "", t)

def diff_bill_formattter(diff_text):
    """returns text with the proposed changes in brackets"""
    # replace ins and dels
    diff_text = diff_text.replace("<ins>", "<INS>")
    diff_text = diff_text.replace("</ins>", "</INS>")
    diff_text = diff_text.replace("<del>", "<DEL>")
    diff_text = diff_text.replace("</del>", "</DEL>")
    # in some states its uppercase
    diff_text = diff_text.replace("<INS>", " {{ ")
    diff_text = diff_text.replace("</INS>", " }} ")
    diff_text = diff_text.replace("<DEL>", " [[ ")
    diff_text = diff_text.replace("</DEL>", " ]] ")
    diff_text = clean_xml(diff_text)

    # Replace multiple whitespaces
    diff_text_list = diff_text.split("\n")
    diff_text_list = [" ".join(sent.split()) for sent in diff_text_list]
    diff_text = "\n".join(diff_text_list)

    # Connect similar contigous changes
    diff_text = diff_text.replace("}} {{", "")
    diff_text = diff_text.replace("]] [[", "")
    diff_text = diff_text.replace("{{ }}", "")
    diff_text = diff_text.replace("[[ ]]", "")

    # Replace multiple whitespaces again
    diff_text_list = diff_text.split("\n")
    diff_text_list = [" ".join(sent.split()) for sent in diff_text_list]
    diff_text = "\n".join(diff_text_list)

    # Keep only End Of Line contiguous to a period (.)
    diff_text = diff_text.replace(". }}\n", ".\n }} ")
    diff_text = diff_text.replace(". ]]\n", ".\n ]] ")
    diff_text_list = diff_text.split(".\n")
    diff_text_list = [sent.replace("\n", " ") for sent in diff_text_list]
    diff_text = ".\n".join(diff_text_list)

    # Connect similar contigous changes again
    diff_text = diff_text.replace("}} {{", "")
    diff_text = diff_text.replace("]] [[", "")
    diff_text = diff_text.replace("{{ }}", "")
    diff_text = diff_text.replace("[[ ]]", "")

    # Add End Of Line at the end of colon (:)
    diff_text = diff_text.replace(":", ":\n")

    # Replace with ins and del
    diff_text = diff_text.replace("{{", "<ins_start>")
    diff_text = diff_text.replace("}}", "<ins_end>")
    diff_text = diff_text.replace("[[", "<del_start>")
    diff_text = diff_text.replace("]]", "<del_end>")

    # Remove visual elements
    diff_text = VISUAL.sub("", diff_text)

    # Return diff text
    return diff_text


def compute_rouge(
    predictions,
    references,
    metrics = ("rouge1", "rouge2", "rougeL")
):
    # Builds rogue scorer and aggregator
    scorer      = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    aggregator  = scoring.BootstrapAggregator()

    # Adds scores to aggregator
    for pred, ref in zip(predictions, references):
        aggregator.add_scores(scorer.score(ref, pred))

    # Aggregates scores
    aggregated = aggregator.aggregate()

    # Builds metrics dict
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = {
            "precision": aggregated[metric].mid.precision,
            "recall":    aggregated[metric].mid.recall,
            "f1":        aggregated[metric].mid.fmeasure,
        }

    return metrics_dict
