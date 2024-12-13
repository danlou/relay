# 📟 Relay: LLMs as IRCs

Please see full details on the [model page](https://huggingface.co/danlou/relay-v0.1-Mistral-Nemo-2407) at HuggingFace.

## How to use

If you have a CUDA GPU (>=12GB VRAM), the best way to use Relay is with the [relaylm.py](https://github.com/danlou/relay/blob/main/relaylm.py) inference script. Just run:
```bash
curl https://danlou.co/f/relaylm.py | python -
```

This script will select the best model for the available VRAM, download, load, and start an interactive chat session.
It does not have any dependencies besides `transformers >= 4.45.1`. You can also download the script manually and then run python, of course.

If you want to use a particular model, you can pass the model name as an argument:
```bash
python relaylm.py danlou/relay-v0.1-Mistral-Nemo-2407-4bit
```

You should see something similar to this demo:
<a href="https://asciinema.org/a/MrPFq2mgIRPruKygYehCbeqwc" target="_blank"><img src="https://asciinema.org/a/MrPFq2mgIRPruKygYehCbeqwc.svg" /></a>

Alternatively, if you do not have a CUDA GPU (e.g., on a Mac), you can use the [GGUF versions](https://huggingface.co/danlou/relay-v0.1-Mistral-Nemo-2407-GGUF) through LM Studio.

With [relaylm.py](https://github.com/danlou/relay/blob/main/relaylm.py), you can also use the model declaratively, outside of an interactive chat session:

```python
from relaylm import suggest_relay_model, RelayLM

def favorite_holiday(relay: RelayLM, country: str) -> str:
    relay.init_context()
    relay.join(role='model', channel=country.lower())
    relay.cast(role='model', desc=f"I'm from {country}.")
    relay.message(role='input', content="What's your favorite holiday?")
    relay.respond(role='model')
    response = relay.get_last()
    return response['content']

model_info = suggest_relay_model()
relay = RelayLM(**model_info)

print(favorite_holiday(relay, 'Portugal'))
print(favorite_holiday(relay, 'China'))
```

## Limitations

This is not a typical AI Assistant. It should perform worse on benchmarks compared to instruct variants.
QLoRa 4bit fine-tuning may be too coarse for preserving integrity of pre-training knowledge.


## License

The model is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
While [Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407) is licensed under Apache 2.0, this Relay fine-tune is trained with a CC-BY-NC 4.0 dataset ([based-chat-v0.1](https://huggingface.co/datasets/danlou/based-chat-v0.1-Mistral-Nemo-Base-2407)).
Code on this repository is Apache 2.0.


## Citation

If you use Relay in your research, please cite it as follows:
```
@misc{relay2024,
  author = {Loureiro, Daniel},
  title = {Relay: LLMs as IRCs},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/danlou/relay}},
}
```
