The file glm_model_new.py builds on Georgi’s original glm_model.py class and introduces some extensions.

- Use of a saved train/validation split, stored in split_indices.npz.
The dataset is divided into training and validation sets only once, and this split is kept to ensure that all evaluations use the same held-out sequences.
  
- The updated model also includes compatibility with HuggingFace’s TrainingArguments, automatically handling differences between older and newer versions (such as evaluation_strategy versus eval_strategy). This ensures that the training code runs across different environments without any manual adjustments.

- After training, model performance is assessed on the validation set. 

- The evaluation computes masked language modeling loss and perplexity, both of which reflect how well the model captures the structure of DNA sequences. 

- The code also introduces a fast variant scoring method called delta_likelihood_fast, which estimates how much less likely a sequence with deletions is compared to the original. 

- see the GLM-2.pdf for more informations. 

- In addition, new dependency and influence mapping tools were developed. These methods track how a change at one position in a sequence affects predictions at other positions. 

- The model itself is trained as a masked-language model, predicting masked nucleotides based on their surrounding context. Since deletions are included in the model’s vocabulary, it also learns how missing bases affect predictions elsewhere in the sequence.


- To better understand these dependencies, a new dependency map method measures the model’s sensitivity to deletions. For a given sequence, a single position is deleted, and the model recomputes predictions for all other positions. The resulting changes are displayed as a heatmap, where each cell represents how much position i influences position j.

- Beyond the modeling itself, the script fundemental_classes/visualization/stats.py provides detailed dataset analysis. It works with the simulated FASTA dataset containing motifs A, B etc... and deletion annotations. By examining FASTA sequence headers, the script determines which motifs are present (both, A only, B only, or none), their start positions, the gap between motifs, and total deletions per sequence. It then counts sequences per motif class, measures the number and position of deletions, and differentiates between deletions occurring before motif A, between motifs A and B, or after motif B. Results are saved as CSV files and visualized in a concise four-panel summary figure showing class frequencies, motif positions, total deletions per sequence, and deletions between motifs.

- The full analysis pipeline can be run as follows:

Train the model with
python fundemental_classes/model_related/train_glm_local.py

- Evaluate model quality using either
python fundemental_classes/model_related/eval_glm_local.py
or
python visualize_data_new.py (which reports validation loss and perplexity).

- Compute dependency maps with
python fundemental_classes/visualization/run_dependency_maps.py
or again via evaluate_display_data.py.

- The dependency analysis prepares validation sequences, separates motif-A and motif-B sequences, computes the maps, and saves all results—heatmaps, input sequences, and manifest CSV files—for further inspection. Finally, dataset-level statistics can be generated using
python fundemental_classes/visualization/stats.py,
which produces the motif and deletion analyses described above.
