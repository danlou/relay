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

print(favorite_holiday(relay, "Portugal"))
print(favorite_holiday(relay, "China"))
