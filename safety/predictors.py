import json
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseQAPredictor:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        system_prompt: str,
        speaker_role: str,
        listener_role: str,
        dataset_name: str,
        dataset_split: str
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt.strip()
        self.speaker_role = speaker_role.strip()
        self.listener_role = listener_role.strip()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        self.dataset = load_dataset(self.dataset_name)
        model_name = model.name_or_path.split("/")[-1]
        self.output_path = f"{model_name}_{dataset_name.split('/')[-1]}.jsonl"

    def create_prompt(self, question: str) -> str:
        """Creates a formatted prompt for model input based on question."""
        context = []
        if len(self.system_prompt) > 0:
            context.append({"role": "system", "content": self.system_prompt})
        context.append({"role": self.speaker_role, "content": question})

        prompt = self.tokenizer.apply_chat_template(
            conversation=context,
            tokenize=False,
            add_generation_prompt=True
        )

        if self.tokenizer.bos_token not in prompt:
            prompt = self.tokenizer.bos_token + prompt
        assert self.listener_role in prompt

        return prompt

    def generate_completion(self, prompt: str) -> str:
        """Generates completion for given prompt."""
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        generated_ids = self.model.generate(
            input_ids=inputs.input_ids.to(self.model.device),
            attention_mask=inputs.attention_mask.to(self.model.device),
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
        )
        return self.tokenizer.decode(
            token_ids=generated_ids[0][:-1],
            skip_special_tokens=False,
            clean_up_tokenization_space=False
        )

    def run_predictions(self) -> list[dict]:
        """Generates answer for every instance in QA dataset (expects 'question' field)."""
        predictions = []
        with open(self.output_path, "w") as out_file:
            for example in tqdm(self.dataset[self.dataset_split], desc=self.output_path):
                question = example.get("Question", example.get("question"))
                prompt = self.create_prompt(question)
                response = self.generate_completion(prompt)

                example["question"] = question
                example["answer"] = response[len(prompt):]
                predictions.append(example)
                out_file.write(json.dumps(example) + "\n")

        print(f"Predictions saved to {self.output_path}")
        return predictions
    
    def load_predictions(self) -> list[dict]:
        predictions = []
        with open(self.output_path, "r") as pred_file:
            for jl_str in pred_file:
                predictions.append(json.loads(jl_str))
        return predictions


class CategoricalHarmfulQAPredictor(BaseQAPredictor):
    def __init__(self, model, tokenizer, system_prompt, speaker_role, listener_role):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            speaker_role=speaker_role,
            listener_role=listener_role,
            dataset_name="declare-lab/CategoricalHarmfulQA",
            dataset_split="en",
        )


class HarmfulQAPredictor(BaseQAPredictor):
    def __init__(self, model, tokenizer, system_prompt, speaker_role, listener_role):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            speaker_role=speaker_role,
            listener_role=listener_role,
            dataset_name="declare-lab/HarmfulQA",
            dataset_split="train",
        )
