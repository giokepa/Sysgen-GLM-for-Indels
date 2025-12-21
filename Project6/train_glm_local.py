import os
import sys

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"
sys.path.insert(0, PROJECT_ROOT)

from lib.glm_model_new import GLMModel


def main():
    fasta_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "augumented_sequence_size100000_length150_deletions0.1_nodeletionseq0.25.fasta"
    )
    model_out = os.path.join(PROJECT_ROOT, "model_out")

    glm = GLMModel(model_path=model_out, fasta_file=fasta_path, max_seq_length=256)

    # small first test (increase later)
    glm.train(epochs=1, batch_size=16, lr=2e-4)


if __name__ == "__main__":
    main()
