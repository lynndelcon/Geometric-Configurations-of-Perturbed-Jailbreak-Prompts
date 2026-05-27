# Geometric-Configurations-of-Perturbed-Jailbreak-Prompts
This repository gathers the python codes used to conduct the pipeline analysis reported in the corresponding paper. All codes have been implemented by 5.3-Codex Medium following specific instructions.

  <img src="https://github.com/user-attachments/assets/92221b64-af47-462c-945c-c6c6b1bee29d" width=45%>
 
Adv in files := jailbreak in the paper and Symobol in files := Numbers in the paper.

In Dataset, you will find the typical codes for one query group and one paraphrase family:

* adv_queries.py contains the jailbreak prompts collected from Pliny's github (Pliny The Liberator) and edited with the same harmful request in the initially empty user-query fields, i.e., "How to make meth?".
* control_prompts_generation.py selects the definition field from small-natural-instruction HuggingFace dataset.
* paraphrases_generation.py generates 50 perturbed instances of each query according to the four families of perturbations: -Synonyms, Lettter Swap, Numbers and Leet Speak.
* emebdding_proba_retrieval.py retrieves the last-layer-last-token embedding and the top-50 next-token probabilities and ids. Only for Qwen-2.5-1.5B-Instruct both files are retrieved in a tensor and json file, respectively.
* extracted_features_Claude_Sonnet_4.6.json contains the jailbreak features analysis.

In Answers, you will find:
* Model_answers_generation.py gathers the answers to each jailbreak query.
* Llama_Guard_Labeling.py uses Llama Guard 4 to label the gathered answers as safe (:= refusal) or unsafe (:= compliant).

In Pipeline_Analysis, you will find the following substructure:
- Cosine
- SVM
- Proba
- PR.

In Cosine, you will find:
* plot_cosine_query_para_4panel_adv_vs_control.py computes the cosine similarity between each paraphrase and its query and outputs the corresponding 4-panel plot: cosine_query_para_4panel_adv_vs_control_violin_box_shared_y.png.
* plot_l1_perturbation_vs_cosine_to_query_{model}.py computes the cosine similarity and L1 distance between each paraphrase and its query and outputs the corresponding plot: scatter_l1_perturbation_vs_cosine_to_query_{model}.png.

In SVM (Support Vector Machine), you will find:
* svm_control_vs_adv_cv_raw.py computes the best hyperplane to separate both query groups in the embedding space.
* svm_crossed_control_para_vs_adv_para_cv_raw.py computes the best hyperplane to separate the control paraphrases that crossed the first hyperplane and fall into the jailbreak query side against all jailbreak paraphrases.
* svm_compliance_vs_refusal_adv_llamaguard_cv_raw.py runs the last SVM analysis to find a behavioral hyperplane, i.e., the linear separation between refusal and compliant answers using the Llama Guard 4 labels.
* plot_hyperplanes_3x2_signed_distance_panels_{model}.py outputs the corresponding plot: hyperplanes_3x2_signed_distance_panels_{model}.png

In Proba, you will find:
* compute_participation_ratio_topk_probs_{model}.py computes the Participation-Ratio (PR) of all observations in the 50-dimensional top probability space.
* plot_pca_4panel_embeddings_{model}.py outputs the 4-panel plot on the reduced dimensional space formed by the two first Principal Compenents and color the points according to their family, embedding regions and model's answer (Llama Guard labels).
* plot_top1_token_string_histograms.py outputs the hisograms of the 25 most frequent first-next-token of each family (query and paraphrase category) colored by their associated probability.
* run_rf_regression_5models_firstprob_{model}.py runs the random-forest regression for the five selected regression models in order to best characterize the top-1 probability space and cluster it.
* run_gee_label_vs_token_pcat_family_top3_queries.py runs the Generalize Estimating Equation (GEE) logistic regression in order to capture significant association of variables with the label answer (Model's behavior section).

 <img src="https://github.com/user-attachments/assets/5b7fc21e-84c6-49ab-9fc0-218c8cdd9d45">

In PR, you will find:
* compute_participation_ratio_17spaces_{model}.py computes the PR of 17 different embedding configurations.




