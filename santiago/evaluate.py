import yaml, torch, pandas as pd
from model import LanguageModel
from utils import compute_rouge
from data_tokenizers import tokenizers_dict

def evaluate_test_set():
    model_path = "model.pth"
    params_path = "params.yml"
    test_csv = "test_set.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading parameters from {params_path}")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    beam_size = params["beam_size_sample"]
    length_penalty = params["length_penalty"]

    print(f"Initializing {params['tokenizer']} tokenizer")
    tokenizer = tokenizers_dict[params["tokenizer"]](
        sequence_raw = [],
        vocab_size = params["vocab_size"],
        max_len = params["max_len"],
        min_summary_length = params["min_summary_length"],
    )

    print(f"Loading model from {model_path}")
    model = LanguageModel(
        tokenizer = tokenizer,
        dim_size = params["dim_size"],
        dim_feedforward = params["dim_feedforward"],
        num_layers = params["num_layers"],
        num_heads = params["num_heads"],
        dropout = params["dropout"],
        max_len = params["max_len"],
        device = device,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)[["diff_text", "summary"]].dropna()
    diffs, refs = df["diff_text"].tolist(), df["summary"].tolist()
    print(f"Found {len(diffs)} test examples")

    print(f"Generating summaries with beam_size={beam_size}, length_penalty={length_penalty}")
    preds = []
    with torch.no_grad():
        for i, diff in enumerate(diffs):
            print(f"Processed {i+1}/{len(diffs)} examples")
            preds.append(model.generate(diff, beam_size, length_penalty))

    print("\nComputing ROUGE scores...")
    rouge = compute_rouge(preds, refs)
    print("ROUGE-1 F1: {:.4f} | ROUGE-2 F1: {:.4f} | ROUGE-L F1: {:.4f}"
          .format(rouge["rouge1"]["f1"],
                  rouge["rouge2"]["f1"],
                  rouge["rougeL"]["f1"]))
    return rouge


# run directly:  python evaluate_test_set.py
if __name__ == "__main__":
    evaluate_test_set()