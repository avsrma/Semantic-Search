import yaml
import json
from typing import List

from sentence_transformers import SentenceTransformer, util


def load_ground_truth(fp="ground_truth_entities.json"):
    """
    :param fp: Path to your dataset containing ground truth names of items (eg. from an API)
    :return: A list of items
    """
    with open(fp, "r", encoding="utf-8") as api_response:
        response = json.load(api_response)
    # process response 
    return response 

def load_synonyms_dict(fp="synonyms.yml"):
    """
    Creates a dictionary of key-items (value of "synonym" key from synonyms.yml) and their corresponding synonyms.
    fp: Path to yml file containing synonyms
    :return: A dictionary containing key-items and corresponding list of synonyms
    """
    with open(fp, encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    syn_dict_keys = [item["synonym"] for item in data["nlu"]]
    syn_dict_values = [item["examples"].split("\n") for item in data["nlu"]]
    syn_dict_values = [[syn.lstrip("- ").strip() for syn in syns if syn != ""] for syns in syn_dict_values]

    return dict(zip(syn_dict_keys, syn_dict_values))


def load_model(model_name="aari1995/German_Semantic_STS_V2"):
    """
    Experimental models
    # "PM-AI/bi-encoder_msmarco_bert-base_german"
    # aari1995/German_Semantic_STS_V2 (current best performing)

    Args:
        model_name: Specify which Transformer models to use

    Returns:
        Instantiated models
    """
    return SentenceTransformer(model_name)


def encode(text_input, model):
    """
    Encodes the input into a higher-dimensional vector space using the transformer models.

    Args:
        text_input: Input string to encode
        model: Already instantiated Transformer models

    Returns:
        Encoded Tensor based on provided Transformer models
    """
    return model.encode(text_input)


def key_item_based_match(key_item, api_items, model, t):
    """
    Computes a match using the key-item from synonyms.yml wrt ground truth items.

    Args:
        key_item: A key-item from synonym.yml 
        api_items: items as present in ground truth
        model: Already instantiated Transformer models
        t: Threshold

    Returns:
        Sorted dictionary of matching item(s) along with confidence
    """

    matches = {}
    emb_input_item = encode(key_item, model)

    for item in api_items:
        emb_item = model.encode(item)
        cosine_similarity = util.cos_sim(emb_input_item, emb_item)
        if cosine_similarity > t:
            matches.update({item: round(cosine_similarity.item(), 2)})
    return dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))


def synonym_cluster_based_match(synonyms_list, api_items, model, t):
    """
    Combines all the synonyms in one cluster and computes similarity wrt each item ground truth. Can return more than
    one match.
    :param synonyms_list: All synonyms of an item  
    :param api_items: items as present in ground truth
    :param model: Already instantiated Transformer model
    :param t: Threshold
    :return: Dictionary of matching item(s) along with confidence
    """
    matches = {}
    synonyms_cluster = " ".join(synonyms_list)
    emb_input_item = encode(synonyms_cluster, model)

    for item in api_items:
        emb_item = encode(item, model)
        confidence = util.cos_sim(emb_input_item, emb_item)
        if confidence > t:
            matches.update({item: round(confidence.item(), 2)})
    return dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))


def semantic_item_matching(model: SentenceTransformer, input_item: str, items: List[str], synonyms: dict,
                               t_key_match=0.85, t_synonym_match=0.8) -> dict:
    """
    
    # Step 1: Match y with ground truth X (eg. API response)
    # Accept match if cosine similarity is above 0.9
    # Else: Continue

    # Step 2: Match the "synonyms cluster" of the item y with X
    # Accept match(es) if cosine similarity is above 0.8
    # Else: No match found

    Args:
        model: Sentence Transformer model
        input_item: item 
        items: ground truth items
        synonyms: List of synonyms (e.g. from synonyms.yml)
        t_key_match: Threshold for Step 1, default 0.85
        t_synonym_match: Threshold for Step 2, default 0.8

    Returns:
        Sorted dictionary of matches
    """
    try:
        matches = key_item_based_match(key_item=input_item,
                                          api_items=items,
                                          model=model,
                                          t=t_key_match)

        return matches if matches else synonym_cluster_based_match(synonyms_list=synonyms[input_item],
                                                                   api_items=items,
                                                                   model=model,
                                                                   t=t_synonym_match)
    except (KeyError, IndexError, TypeError):
        return {}


# model = load_model()
# synonyms = load_synonyms_dict("synonyms.yml")

# input_item = "akute Schmerzen"
# items = [] # add ground truth
# Step 1: Key-item based match
# print("Key-item based match: ", key_item_based_match("input_item", items, model, t=0.5))

# Step 2: Synonym-cluster based match
# print("Synonym cluster based match: ", synonym_cluster_based_match(synonyms["Haarsprechstunde"], items, model))

# Run experiment to loop over all key-items
# for k in synonyms.keys():
#     print(f"Key-item based match, key-item: {k} ", key_item_based_match(k, items, model))
#     print(f"Synonym cluster based match, key-item: {k} ", synonym_cluster_based_match(synonyms[k], items, model))
