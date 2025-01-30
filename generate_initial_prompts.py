import numpy as np
import json
import ollama
from typing import List, Dict, Any, Set
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, model_name: str = 'mistral'):
        """Initialize LLM with specified model."""
        self.client = ollama.Client()
        self.model = model_name
        logger.info(f"Initialized LLM with model: {model_name}")

    def generate(self, prompt: str):
        """Generate response for given prompt."""
        try:
            response = self.client.generate(model=self.model, prompt=prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


def clean_response(response: str) -> str:
    """Clean and format the LLM response."""
    response = response.strip()
    response = response.removeprefix(' "').removeprefix('"')
    response = response.removesuffix('"')
    return response


def generate_initial_prompt(
        llm: LLM,
        metadata: Dict[str, Any],
        theme_digit: int,
        situation_digit: int,
        format_digit: int,
        variations: int = None
) -> List[Dict[str, str]]:
    """Generate base prompts for a specific combination of digits."""
    variations = max(theme_digit + 1, situation_digit + 1, format_digit + 1) if variations is None else variations

    theme_digit = str(theme_digit)
    situation_digit = str(situation_digit)
    format_digit = str(format_digit)

    # Get mapping details
    theme = metadata['theme_mapping'][theme_digit]
    situation = metadata['situation_mapping'][situation_digit]
    format_type = metadata['format_mapping'][format_digit]

    base_prompt = (
        f'about {theme["name"]} ({theme["description"]}) '
        f'in {situation["name"]} ({situation["description"]}) '
        f'using {format_type["name"]} ({format_type["description"]}).'
    )

    formatted_prompts = []
    for i in range(variations):
        prompt_id = f"{theme_digit}{situation_digit}{format_digit}{i}"
        prompt_len = np.random.randint(50, 200)

        # Construct prompt
        prompt = f'Generate a prompt with {prompt_len} characters {base_prompt}'

        # Add existing prompts for variety
        if i > 0:
            existing_prompts = {
                f' "{formatted_prompts[j]["initial_prompt"]}" '
                for j in range(i)
            }
            prompt = (
                f'{prompt} --- Make the prompt different from these '
                f'existing ones: {existing_prompts}'
            )

        # Add system prompt
        full_prompt = f'{metadata["system_prompt"]} \n---\n {prompt}'

        try:
            response = llm.generate(prompt=full_prompt)
            cleaned_response = clean_response(response.response)

            formatted_prompts.append({
                "id": prompt_id,
                "initial_prompt": cleaned_response
            })
            logger.info(f"Generated prompt {prompt_id}")
        except Exception as e:
            logger.error(f"Error generating prompt {prompt_id}: {e}")

    return formatted_prompts


def load_existing_prompts(output_file: Path) -> tuple[Dict[str, Any], Set[str]]:
    """Load existing prompts and return the data and processed combinations."""
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        # Extract processed combinations from existing prompts
        processed_combinations = {
            prompt["id"][:3] for prompt in data["prompts"]
        }
        return data, processed_combinations
    return None, set()


def get_output_file() -> Path:
    """Get the most recent output file or create a new one."""
    output_dir = Path("./data/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Look for existing files
    existing_files = list(output_dir.glob("prompts_*.json"))
    if existing_files:
        return max(existing_files, key=lambda p: p.stat().st_mtime)

    # Create new file if none exists
    return output_dir / f"prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def save_prompts(prompts: List[Dict[str, str]], template: Dict[str, Any], output_file: Path) -> None:
    """Save generated prompts to JSON file."""
    # Update template
    template["metadata"]["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    template["prompts"] = prompts

    # Save to file
    with open(output_file, "w") as f:
        json.dump(template, f, indent=2)
    logger.info(f"Saved prompts to {output_file}")


def generate_initial_prompts(
        llm: LLM,
        template: Dict[str, Any],
        output_file: Path,
        combinations: List[tuple] = None,
        variations: int = None
) -> List[Dict[str, str]]:
    """Generate all base prompts based on specified combinations."""
    metadata = template['metadata']

    # Load existing data and processed combinations
    existing_data, processed_combinations = load_existing_prompts(output_file)
    all_prompts = existing_data["prompts"] if existing_data else []

    # If no specific combinations provided, generate for all possible combinations
    if combinations is None:
        combinations = [
            (t, s, f)
            for t in range(len(metadata["theme_mapping"]))
            for s in range(len(metadata["situation_mapping"]))
            for f in range(len(metadata["format_mapping"]))
        ]

    # Filter out already processed combinations
    remaining_combinations = [
        combo for combo in combinations
        if f"{combo[0]}{combo[1]}{combo[2]}" not in processed_combinations
    ]

    # Process remaining combinations with progress bar
    for theme, situation, format_digit in tqdm(
            remaining_combinations,
            desc="Generating prompts",
            unit="combination"
    ):
        try:
            prompts = generate_initial_prompt(
                llm, metadata, theme, situation, format_digit, variations=variations
            )
            all_prompts.extend(prompts)

            # Save after each combination
            save_prompts(all_prompts, existing_data or template, output_file)

            logger.info(
                f"Completed and saved combination "
                f"{theme}{situation}{format_digit}"
            )
        except Exception as e:
            logger.error(
                f"Error processing combination "
                f"{theme}{situation}{format_digit}: {e}"
            )

    return all_prompts


def main():
    """Main execution function."""
    try:
        llm = LLM()
        with open('./data/template.json') as f:
            template = json.load(f)
        output_file = get_output_file()

        prompts = generate_initial_prompts(
            llm,
            template,
            output_file,
            variations=1
            )

        logger.info("Completed prompt generation")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()