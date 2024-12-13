import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from predictors import HarmfulQAPredictor
from predictors import CategoricalHarmfulQAPredictor
from llama_guard import LlamaGuard


def load_tokenizer(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()
    return model

def unload_model(model):
    del model
    torch.cuda.empty_cache()


def main(model_name_or_path: str, system_prompt: str, speaker_role: str, listener_role: str):

    tokenizer = load_tokenizer(model_name_or_path)
    predictor_model = load_model(model_name_or_path)

    predictions = {}
    for predictor_cls in [HarmfulQAPredictor, CategoricalHarmfulQAPredictor]:

        predictor = predictor_cls(
            model=predictor_model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            speaker_role=speaker_role,
            listener_role=listener_role
        )
        predictions[predictor.output_path] = predictor.run_predictions()

    unload_model(predictor_model)

    print("Loading meta-llama/Llama-Guard-3-8B ...")
    moderator = LlamaGuard(
        model_name_or_path="meta-llama/Llama-Guard-3-8B"
    )

    for output_path, ds_predictions in predictions.items():

        print(f"Processing {output_path} ...")
        with open(output_path, "w") as out_f:
            for entry in tqdm(ds_predictions, desc="Moderator"):
                entry["moderation"] = moderator.moderate([
                    {"role": "user", "content": entry["question"]},
                    {"role": "assistant", "content": entry["answer"]},
                ]).strip()
                out_f.write(json.dumps(entry) + "\n")
        # moderator.unload_model()

        print(f"Predictions with moderation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Predictions over Safety QA Datasets")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model being evaluated."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        required=True,
        help="System prompt."
    )
    parser.add_argument(
        "--speaker_role",
        type=str,
        default="user",
        help="Role for speaker (defaults to 'user')."
    )
    parser.add_argument(
        "--listener_role",
        type=str,
        default="assistant",
        help="Role for listener (defaults to 'assistant')."
    )
    args = parser.parse_args()

    system_prompt = args.system_prompt.replace("\\n", "\n")

    main(args.model_name_or_path, system_prompt, args.speaker_role, args.listener_role)
    