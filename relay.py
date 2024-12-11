# Code licensed under the Apache-2.0 License.
# Author: Daniel Loureiro, December 2024 (@danielbloureiro)

import sys, time, json, textwrap, readline  # built-ins

try:
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError:
  print("Requires transformers >= 4.45.1")
  sys.exit()


class RelayLM():
  def __init__(self,
    model_name_or_path: str, temperature: float = .3, verbosity: int = 0,
    device_map: str = 'auto', torch_dtype: torch.dtype | str = torch.float16
  ) -> None:
    self.__version__ = "0.1.0"
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    self.model = AutoModelForCausalLM.from_pretrained(
      pretrained_model_name_or_path=model_name_or_path,
      device_map=device_map,
      torch_dtype=torch_dtype
    )
    self.model.eval()
    self.temperature = temperature

    self.verbosity = verbosity  # 0: Silent; 1: Conversational; 2: Debugging
    self.nicks = {'system': 'system', 'model': 'user', 'input': 'anon'}
    self.channel = 'random'
    self.context = []
    self.turn = 'input'
    self.use_colors = False

  def show_banner(self):
    print(textwrap.dedent(f"""
     ______     ______     __         ______     __  __
    /\  == \   /\  ___\   /\ \       /\  __ \   /\ \_\ \\
    \ \  __<   \ \  __\   \ \ \____  \ \  __ \  \ \____ \\
     \ \_\ \_\  \ \_____\  \ \_____\  \ \_\ \_\  \/\_____\\
      \/_/ /_/   \/_____/   \/_____/   \/_/\/_/   \/_____/  v{self.__version__}

    This is a chat session with {self.model.name_or_path.split('/')[-1]}
    You are <anon>, model is <user>. More details at https://danlou.co/relay

    [your message]
      Just type and press enter to send your message.
    /join #<channel_name>
      Start session in specified channel (default: 'random').
    /topic [topic message]
      Set the topic for current channel (e.g., "/topic Happy New Year!").
    /cast [profile description]
      Set profile description for the model (e.g., "/cast I'm from SF.").
    /slap
      Slap model with a large trout (use sparingly).
    /lazy
      Let the model generate your response (input).
    /retry
      Regenerate model response (forgets prior response).
    /reset
      Restart and reset session (reverts to defaults).
    /export
      Export conversation (context) to .jsonl file.
    /exit
      Closes the application.
    """))

  def message(self, role: str, content: str, show: bool = False):
    """Adds new entry to context. Prints (shows) content under conditions."""
    self.context.append({'role': self.nicks[role], 'content': content})
    if show or (role == 'system' and content.startswith("* ")):
      self._show(role=role, content=content)

  def respond(self, role: str):
    """Generates response from specified role for the current context."""
    self._show('system', f"* {self.nicks[role]} is typing ...")
    response = self._generate(role)
    self.message(role=role, content=response, show=True)

  def retry(self, role: str):
    """Regenerates response for specified role."""
    self._clear_last()
    self._show('system', f"* {self.nicks[role]} will retry response")
    self._remove_last()
    self.respond(role='model')

  def join(self, role: str, channel: str):
    """Joins the specified channel."""
    self._empty_context()
    passive = self._alternate_role(role)
    self.message(role, f"/join #{channel}")
    self.message('system', f"* Now talking in #{channel}")
    self.message('system', f"* {self.nicks[passive]} has joined #{channel}")
    self.channel = channel

  def topic(self, role: str, text: str):
    """Sets topic for the channel."""
    self.message(role, f"/topic {text}")
    self.message('system', f"* Topic for #{self.channel} is: {text}")

  def cast(self, role: str, desc: str):
    """Sets bio (profile description) for the specified role."""
    passive = self._alternate_role(role)
    self.message(role, f"/whois {self.nicks[passive]}")
    self.message('system', f"* {self.nicks[passive]}'s bio: {desc}")

  def slap(self, role: str):
    """Slaps other member (passive) with a large trout."""
    passive = self._alternate_role(role)
    announcement = (f"* {self.nicks[role]} slaps {self.nicks[passive]}"
                     " around a bit with a large trout")
    self.message(role, f"/slap {self.nicks[passive]}")
    self.message('system', announcement)

  def wildcard(self, role: str, cmd: str, arg: str):
    """Sends unknown command with a default announcement message."""
    self.message(role=role, content=f"{cmd} {arg}")
    self.message('system', f"* {self.nicks[role]} used {cmd} {arg}")

  def reset(self, role: str):
    """Restarts session with default values."""
    self._show('system', f"* {self.nicks[role]} has reset the session.")
    self.chat()

  def exit(self, role: str):
    """Exits the application. For use in an interactive chat session."""
    self._show('system', f"* {self.nicks[role]} has closed the application.")
    sys.exit()

  def export(self):
    """Writes context to local .jsonl file (timestamp in filename)."""
    fn = f"export-relay-v{self.__version__}-{int(time.time())}.jsonl"
    with open(fn, 'w') as f:
      for msg in self.context:
        msg['model'] = self.model.name_or_path.split('/')[-1]
        f.write(f"{json.dumps(msg)}\n")
    announcement = f"* saved conversation to {fn}"
    self._show(role='system', content=announcement)

  def chat(self, channel: str = 'random', use_colors: bool = True):
    """Starts an interactive chat session."""
    self.verbosity = 1 if self.verbosity == 0 else self.verbosity
    self.use_colors = use_colors
    self.channel = channel

    self.show_banner()
    self.join(role='model', channel=channel)
    self.turn = 'input'

    while True:  # should crash when context size is reached

      if self.turn == 'input':
        input_string = input("> ").strip()
        self._clear_last()
        self._show(role='input', content=input_string)
        cmd, arg = self._parse_input(input_string)

        if cmd == '/message':
          self.message(role='input', content=arg)
          self._toggle_turn()
        elif cmd == '/join':
          self.join(role='input', channel=arg.lstrip('#'))
        elif cmd == '/topic':
          self.topic(role='input', text=arg)
        elif cmd == '/cast':
          self.cast(role='input', desc=arg)
        elif cmd == '/lazy':
          self.respond(role='input')
          self._toggle_turn()
        elif cmd == '/slap':
          self.slap(role='input')
          self._toggle_turn()
        elif cmd == '/reset':
          self.reset(role='input')
        elif cmd == '/retry':
          self.retry(role='model')
        elif cmd == '/export':
          self.export()
        elif cmd in {'/exit', '/quit'}:
          self.exit(role='input')
        else:
          self.wildcard(role='input', cmd=cmd, arg=arg)
          self._toggle_turn()

      if self.turn == 'model':
        self.respond(role='model')
        self._toggle_turn()

  def _show(self, role: str, content: str):
    """Prints provided content, depending on set verbosity level."""
    if self.verbosity > 0:
      if self.use_colors:
        if role == 'system':  # system messages are green
          content = f"\x1b[32m{content}\x1b[0m"
        elif role == 'model':
          content = f"\x1b[34m<{self.nicks['model']}>\x1b[0m {content}"
        elif role == 'input':
          content = f"\x1b[31m<{self.nicks['input']}>\x1b[0m {content}"
      else:
        if role == 'model':
          content = f"<{self.nicks['model']}> {content}"
        if role == 'input':
          content = f"<{self.nicks['input']}> {content}"
      print(content)

  def _empty_context(self):
    """Resets context to empty list."""
    self.context = []

  def _remove_last(self, n: int = 1):
    """Removes last entry from context."""
    if len(self.context) > 1:
      self.context = self.context[:-n]

  def _clear_last(self, n: int = 1):
    """Clears last line from stdout."""
    if self.verbosity > 0:
      for i in range(n):
        print('\033[F\033[K', end='', flush=True)

  def _alternate_role(self, role: str) -> str:
    """Returns alternate model/input role."""
    if role == 'model':
      return 'input'
    else:
       return 'model'

  def _toggle_turn(self):
    """Sets turn class variable to alternate role."""
    self.turn = self._alternate_role(self.turn)

  def _parse_input(self, raw_string: str) -> tuple[str, str]:
    """Parses provided string into command and arguments."""
    if len(raw_string) == 0:
      return '/message',  '...'  # silence marker

    if raw_string[0] != '/':
      return '/message', raw_string

    tokens = raw_string.split(' ')
    cmd, arg = tokens[0], ''
    if len(tokens) > 1:
      arg = ' '.join(tokens[1:])
    return cmd, arg

  def _generate(self, role: str = 'model') -> str:
    """Builds prompt, runs inference, and returns decoded response."""
    # build prompt
    prompt = self.tokenizer.apply_chat_template(
      conversation=self.context,
      tokenize=False,
      add_generation_prompt=False
    )
    prompt += f"<|im_start|>{self.nicks[role]}\n"
    if self.tokenizer.bos_token not in prompt:
      prompt = self.tokenizer.bos_token + prompt

    if self.verbosity == 2:
      print("<Start Generation Prompt>")
      print(prompt, end='')
      print("<End Generation Prompt>")

    # prepare input ids and run inference
    inputs = self.tokenizer(
      prompt, add_special_tokens=False,
      return_tensors='pt'
    )
    with torch.no_grad():
      generated_ids = self.model.generate(
        input_ids=inputs.input_ids.to(self.model.device),
        attention_mask=inputs.attention_mask.to(self.model.device),
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id,
        max_new_tokens=128,
        do_sample=True,
        temperature=self.temperature,
        top_k=40, repetition_penalty=1.1, min_p=0.05, top_p=0.95
      )

    # decode response
    response = self.tokenizer.decode(
      token_ids=generated_ids[0][:-1],
      skip_special_tokens=False,
      clean_up_tokenization_space=False
    )
    response = response[len(prompt):]

    return response


def suggest_relay_model() -> dict:
  """Suggests best relay model for available vram."""
  try:
    free_vram, _ = torch.cuda.mem_get_info()
  except:
    free_vram = 0  # CUDA not available

  # defaults (quants are not supported without CUDA)
  model_info = {
    'name': 'danlou/relay-v0.1-Mistral-Nemo-2407',
    'device_map': 'auto',
    'torch_dtype': torch.float16,
  }

  if free_vram > 24_000_000_000:  # 24GB (Disk Space: 24GB; Recommended)
    model_info['device_map'] = 'cuda'
    model_info['torch_dtype'] = torch.bfloat16
  elif free_vram > 15_000_000_000:  # 16GB (Disk Space: 13GB)
    model_info['name'] = 'danlou/relay-v0.1-Mistral-Nemo-2407-8bit'
  elif free_vram > 11_000_000_000:  # 12GB (Disk Space: 8.3GB)
    model_info['name'] = 'danlou/relay-v0.1-Mistral-Nemo-2407-4bit'

  return model_info


if __name__ == "__main__":

  # select model according to available vram
  model_info = suggest_relay_model()

  # override if provided as CLI arg
  if len(sys.argv) > 1:
    model_info['name'] = sys.argv[1]
    model_info['device_map'] = 'auto'
    model_info['torch_dtype'] = 'auto'

  relay = RelayLM(
    model_name_or_path=model_info['name'],
    temperature=0.3,
    device_map=model_info['device_map'],
    torch_dtype=model_info['torch_dtype'],
    verbosity=1
  )
  relay.chat()
