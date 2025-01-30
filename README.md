# UnlockAI: Anti-Speciesist Constitutional AI

This repository contains the implementation of an exploratory study on using Constitutional AI techniques to address value lock-in in Large Language Models (LLMs), with a specific focus on reducing speciesist bias. The project demonstrates a proof-of-concept pipeline for generating training data that could help align language models with vegan ethical principles.

## Project Overview

Value lock-in occurs when current societal values become embedded and potentially perpetuated through AI systems. This project explores how Constitutional AI methodology might be adapted to implement alternative ethical frameworks, using speciesist bias as a case study.

The repository includes:
- `Vegan_Constitutional_AI.ipynb`: Main Jupyter notebook containing the dataset generation pipeline
- `generate_initial_prompts.py`: Script for synthetic initial prompt generation (sequential processing)
- `generate_initial_prompts_parallel.py`: Parallel version of the prompt generation script
- `/data`: Example outputs and initial redteaming prompts (from both Mistral and Claude)
- `UnlockAI__Addressing_Value_Lock_in_Through_Anti_Speciesist_Constitutional_AI.pdf`: Accompanying research paper

## Pipeline Components

The dataset generation pipeline consists of three main stages:
1. Initial response generation to potentially speciesist prompts
2. Critique based on vegan ethical principles
3. Response revision to align with anti-speciesist values

## Requirements

- [Ollama](https://ollama.com/) installed
- Mistral 7B model downloaded through Ollama
- Python 3.8+
- Required Python packages:
  - ollama
  - numpy
  - jupyter

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/veganCAI.git
```

2. Ensure Ollama is running and Mistral 7B is installed:
```bash
ollama pull mistral
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook Vegan_Constitutional_AI.ipynb
```

Note: The pipeline can be easily adapted to use any open-source model provided by Ollama by replacing "mistral" with the desired model name in the LLM class instantiation.

## Project Context

This project was developed as part of the AI Safety Fundamentals course on AI Alignment by BlueDot Impact (https://aisafetyfundamentals.com/alignment/). It explores practical approaches to AI alignment while considering how we might develop AI systems that can adapt to moral progress rather than permanently encoding current ethical perspectives.

## License

This project is licensed under the MIT License.

## Acknowledgments

This project was completed as my final project for the [AI Alignment course](https://aisafetyfundamentals.com/alignment/) by BlueDot Impact.