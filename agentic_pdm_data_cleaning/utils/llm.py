from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
import yaml
from agentic_pdm_data_cleaning.utils.filesystem import FileSystem


def get_model(model_name: str = "llama3.1:latest") -> OpenAIModel:
    fs = FileSystem()
    model_config_path = fs.model_config_file(model_name)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_name}")
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    model_name = model_config.get("full_model_identifier", "llama3.1:latest")
    llm_section = model_config.get("llm_config", {})
    provider_type = llm_section.get("provider", "open_ai")
    if provider_type == "open_ai":
        # Use OpenAIProvider for OpenAI models
        provider = OpenAIProvider(base_url=llm_section.get(
            "base_url", "https://api.openai.com/v1"))
    elif provider_type == "open_router":
        # Use OpenRouterProvider for OpenRouter models
        provider = OpenRouterProvider()
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    # Define the model
    print(f"Loading model: {model_name} from provider: {provider_type}")
    model = OpenAIModel(
        model_name=model_name,
        provider=provider,
    )
    model.model_name
    return model


class CostManager:
    def __init__(self):
        fs = FileSystem()
        self.pricing = {}
        for config_file in fs.models_config_dir.glob("*.yaml"):
            with open(config_file, 'r') as f:
                model_config = yaml.safe_load(f)
            model_name = model_config.get("model_name", "")
            llm_section = model_config.get("llm_config", {})
            input_tokens_cost = llm_section.get("cost_in", 0.14)
            output_tokens_cost = llm_section.get("cost_out", 1.4)
            unit = llm_section.get("cost_unit", 1_000_000)
            self.pricing[model_name] = {
                "in": input_tokens_cost,
                "out": output_tokens_cost,
                "unit": unit
            }

    def get_model_costs(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        if model_name not in self.pricing:
            return 0.0
        p = self.pricing[model_name]
        cost = (input_tokens * p["in"] / p["unit"]) + \
            (output_tokens * p["out"] / p["unit"])
        return float(cost)


if __name__ == "__main__":
    model = get_model("llama3.1:latest")
    print(model.model_name)
    print(model.base_url)
    print(model)
