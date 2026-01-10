import torch

from cs336_basics.config import ModelConfig
from cs336_basics.generate import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer.tokenizer import load_tokenizer_from_dir
from cs336_basics.utils import get_device, print_color

if __name__ == "__main__":
    # Example usage
    prompt = "Once upon a time"
    checkpoint_dir = "checkpoints/tiny_stories_transformer/best_model_step_5000.pt"

    device = get_device(verbose=False)
    model_config = ModelConfig()
    model = TransformerLM(model_config).to(device)
    states = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(states["model_state_dict"])
    tokenizer = load_tokenizer_from_dir("./datasets/tiny_stories")

    print("Generating text...")
    generated_outputs = generate(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=512,
        top_p=0.5,
        temperature=0.8,
    )
    generated_text = generated_outputs["generated_text"]
    print("Once upon a time", end="")
    print_color(f"{generated_text}\n", "cyan")
