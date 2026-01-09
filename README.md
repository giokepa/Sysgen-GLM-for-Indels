Amelie Part

The file glm_model_new.py is based on Georgi’s original glm_model.py class and extends it with several important additions. 

These extensions include a saved train/validation split stored in split_indices.npz, compatibility with different HuggingFace TrainingArguments versions (handling evaluation_strategy versus eval_strategy), an evaluation routine for masked-language-model quality on the validation set (computing loss and perplexity with CSV export), a fast variant scoring method called delta_likelihood_fast, and an influence-based method called influence_probability_shift.

A key extension is the saved train/validation split. 

The dataset is split only once into training and validation sets, and this split is stored on disk. 

This ensures that all evaluations are performed on exactly the same held-out sequences, making results reproducible and directly comparable across runs.

Another extension concerns HuggingFace compatibility. 


The training code is written to work across different HuggingFace versions, which allows the model to run reliably on different machines and environments without manual changes.

Model quality is assessed on the validation set after training. The model is evaluated on the full set of validation sequences using masked-language-model loss and perplexity. These two metrics together provide a clear, global measure of how well the model captures the structure of the DNA sequences.
The method delta_likelihood_fast enables fast variant scoring. It measures how much a sequence containing deletions becomes less likely under the model compared to the original sequence. This provides a direct and interpretable way to score insertion and deletion variants.

In addition, dependency and influence mapping methods are implemented. These track how changes at one position in a sequence affect the model’s predictions at other positions. This makes it possible to visualize long-range dependencies and motifs learned by the model.

The model itself is trained as a masked-language model. During training, it predicts a masked nucleotide based on the surrounding sequence context. Because deletions are explicitly included as part of the model’s vocabulary, the model learns how missing bases influence predictions at other positions. This makes it possible to study motif structure, long-range dependencies, and the way deletions propagate information through DNA sequences.
During training, the dataset is split once into training and validation sets and the split is saved. All later evaluations are therefore carried out exclusively on held-out validation sequences. After training, a model quality score is computed on the validation set using masked-language-model loss and perplexity. Together, these provide a single, objective summary of how well the GLM understands DNA.


To analyze dependencies learned by the model, a new dependency map method was added that measures sensitivity to deletions. For a given sequence, a single position is deleted, the predicted base distributions at all other positions are recomputed, and the resulting changes are stored as a heatmap. Each cell in this heatmap answers the question: “If position i is deleted, how much does it affect position j?” These maps reveal motif boundaries, structural regions, and long-range dependencies in an intuitive way.
In addition to the model code, the script fundemental_classes/visualization/stats.py (which already acts as a main script and can be run directly) analyzes the simulated FASTA dataset containing motif A, motif B, and deletion annotations. It reads FASTA files whose sequence headers encode which motifs are present (both, A_only, B_only, or no_motif), the start positions of motif A and motif B, the gap between motifs, and the total number of deletions in each sequence. The script counts how many sequences belong to each motif class, computes how many deletions occur per sequence, and measures where deletions occur relative to the motifs, distinguishing deletions before motif A, between motif A and motif B, and after motif B. It outputs CSV files containing per-sequence statistics for each motif class, as well as a single four-panel summary figure showing class frequencies, motif positions, total deletions per sequence, and deletions between motif A and motif B.

To run the full pipeline, the model can first be trained using
python fundemental_classes/model_related/train_glm_local.py.
Model quality can then be evaluated with
python fundemental_classes/model_related/eval_glm_local.py,
or alternatively by running
python visualize_data_new.py,
which produces validation loss and perplexity.

Dependency maps can be computed by running
python fundemental_classes/visualization/run_dependency_maps.py,
or again via
python visualize_data_new.py.

This selects clean validation sequences, separates motif-A and motif-B sequences, computes dependency heatmaps, and saves all maps together with the input sequences and a manifest CSV file.
Finally, dataset statistics can be generated by running
python fundemental_classes/visualization/stats.py,
which produces the motif and deletion analyses described above.

