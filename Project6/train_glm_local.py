import os
from glm_model_new import GLMModel

PROJECT_ROOT = "/Users/amelielaura/Documents/Project6"

FASTA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "augumented_sequence_size100000_length150_deletions0.1_nodeletionseq0.25.fasta"
)
MODEL_OUT = os.path.join(PROJECT_ROOT, "model_out")

def main():
    glm = GLMModel(model_path=MODEL_OUT, fasta_file=FASTA_PATH, max_seq_length=256)
    glm.train(epochs=2, batch_size=16, lr=2e-4)  # start small for local test
    print("Done training. Model saved in:", MODEL_OUT)

if __name__ == "__main__":
    main()
