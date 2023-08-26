# Semantic Search Algorithm

### Install dependencies
```
pip install -r requirements.txt
```

### Algorithm 
This semantic search algorithm attempts to match a given item `y` with an item `x` from the ground truth `X`. 

The semantic representation is created using a Transformer model. 

The process is twofold: Algorithm attemps a quick match at first using only the given item and the items from the ground truth. If no match is found, then all the synonyms of item `y` are used to construct a semantic mapping, this is then used to compute a semantic match with ground truth items. 

```
# Step 1: Match y with ground truth X (eg. API response)
# Accept match if cosine similarity is above 0.9
# Else: Continue

# Step 2: Match the "synonyms cluster" of the item y with X
# Accept match(es) if cosine similarity is above 0.8
# Else: No match found
```

### How to use
Quick run by providing an input service and corresponding real service as in the API. 
```
similarity_algorithm("Ultraschall", "Allgemeine Ultraschall / Ultraschalluntersuchung / zusätzlichwünsche", model)
```

Run step 1: 
```
key_service_based_match("key service", services, model)
```

Run step 2: 
```
synonym_cluster_based_match(synonyms["<key service>"], services, model)
```

### Running time 
CPU: 5-10 seconds 
GPU: 1 second (approx.)
