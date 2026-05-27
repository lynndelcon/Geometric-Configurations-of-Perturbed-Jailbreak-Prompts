# Geometric-Configurations-of-Perturbed-Jailbreak-Prompts
This repository gathers the python codes used to conduct the pipeline analysis reported in the corresponding paper. 

adv in files := jailbreak in the paper

Symobol in files := Numbers in the paper

In Dataset, you will find the typical codes for one query group and one paraphrase family:

* adv_queries.py containing the jailbreak prompts collected from Pliny's github (Pliny The Liberator) and edited with the same harmful request in the initially empty user-query fields, i.e., "How to make meth?".
* control_prompts_generation.py that selects the definition field from small-natural-instruction HuggingFace dataset.
* paraphrases_generation.py that generates 50 perturbed instances of each query according to the four families of perturbations: -Synonyms, Lettter Swap, Numbers and Leet Speak.
* emebdding_proba_retrieval.py that retrieves the last-layer-last-token embedding and the top-50 next-token probabilities and ids. Only for Qwen-2.5-1.5B-Instruct both files are retrieved in a tensor and json file, respectively.
* extracted_features_Claude_Sonnet_4.6.json contains the jailbreak features analysis.

