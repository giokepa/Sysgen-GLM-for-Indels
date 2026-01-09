Amelie Part

glm_model_new.py based on Georgi´s glm_model.py class + extensions:

   Extensions:
     - train/val split saved to split_indices.npz
     - HF TrainingArguments compatibility (evaluation_strategy vs eval_strategy)
     - evaluate_mlm_quality_on_val (loss + perplexity on VAL) + CSV export
     - delta_likelihood_fast
     - influence_probability_shift


Extensions

Saved train/validation split

The dataset is split once into train and validation and saved. This makes all evaluations better and ensures we always test on the same sequences.


HuggingFace compatibility

The training code works across different HuggingFace versions, so it runs reliably on different machines.

Model quality on validation set

After training, the model is evaluated on our whole fasta file sequences using loss and perplexity, giving a clear overall quality score.


Fast variant scoring (delta_likelihood_fast)

Measures how much a sequence with deletions becomes less likely than the original — a direct way to score indels.


Dependency and influence mapping

Tracks how changes at one position affect predictions at other positions, letting us visualize long-range dependencies and motifs learned by the model.


What the model does


The model is trained as a masked-language model: it predicts a masked nucleotide from the surrounding sequence.

Because deletions are part of the vocabulary, the model learns how missing bases affect predictions elsewhere in the sequence.


This makes it possible to study:

• motif structure

• long-range dependencies

• and how deletions propagate information through DNA


How training and validation work


When we train the model, the dataset is split into train and validation once and the split is saved.

This ensures all later evaluations are done on held-out sequences.


After training, we compute a model quality score on the validation set:

• masked-language-model loss

• and perplexity


This gives a single, objective number that summarizes how well the GLM understands DNA.

What the dependency maps are

We added a new dependency map method that measures how sensitive the model is to deletions.

For a given sequence:


We delete one position
We check how the predicted base distributions change at all other positions
We store this as a heatmap

Each cell answers:


“If position i is deleted, how much does it affect position j?”

This reveals motif boundaries, structural regions and long-range dependencies in a good way.

And then fundemental_classes/visualization/stats.py (already the main.py and run it like that)

This script analyzes the simulated FASTA dataset containing motif A, motif B, and deletion annotations.

It reads the FASTA file where each sequence header encodes:
	•	which motifs are present (both, A_only, B_only, no_motif)
	•	the start positions of motif A and motif B
	•	the gap between motifs
	•	the total number of deletions in the sequence

The script then:
	•	counts how many sequences belong to each motif class
	•	computes how many deletions occur in each sequence
	•	measures where deletions occur relative to motifs
(before motif, after motif, and between motif A and B)

It outputs:
	•	CSV files with per-sequence statistics for each motif class
	•	a single four-panel summary figure showing
	1.	class frequencies,
	2.	motif positions,
	3.	total deletions per sequence, and
	4.	deletions between motif A and motif B

How to run everything

Train the model
python fundemental_classes/model_related/train_glm_local.py
and then evaluate model quality with python fundemental_classes/model_related/eval_glm_local.py
or only: python visualize_data_new.py

This produces validation loss and perplexity.

Run dependency maps

python fundemental_classes/visualization/run_dependency_maps.py or only python visualize_data_new.py

This:

• selects clean validation sequences

• separates motif-A and motif-B sequences

• computes dependency heatmaps

• saves all maps, input sequences and a manifest CSV

And then fundemental_classes/visualization/stats.py (already the main.py and run it like that)


