import os
import argparse
from glm_model_new import GLMModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--model_out", default="model_out")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)         # start small locally
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    os.makedirs(args.model_out, exist_ok=True)

    glm = GLMModel(model_path=args.model_out, fasta_file=args.fasta, max_seq_length=args.max_len)
    glm.train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)

if __name__ == "__main__":
    main()
