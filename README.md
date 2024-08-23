# MetaAgentFive

MetaAgentFive is a metacognitive reasoning agent that uses OpenAI's GPT models to analyze input, generate insights, and provide structured responses based on various reasoning templates.

## Features

- Process input data through a series of analytical questions
- Generate class labels based on the analysis
- Choose appropriate reasoning templates
- Fill templates with context-specific information
- Provide final completions with summaries, evaluations, and suggestions
- CLI support for easy interaction

## Requirements

- Python 3.7+
- `openai` Python package
- OpenAI API compat chat service (I use VLLM locally).

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/MetaAgentFive.git
   cd MetaAgentFive
   ```

2. Install the required packages:
   ```
   pip install openai
   ```

3. Set up your OpenAI API key and URL as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   export OPENAI_BASE_URL="http://0.0.0.0:8000/v1"
   ```

## Usage

Run the script from the command line with the following options:

```
python MetaAgentFive.py --input <input_string_or_file> [--context <context_string_or_file>]
```

- `--input`: Required. Specify either an input string or a path to a file containing the input.
- `--context`: Optional. Provide additional context as a string or a path to a file containing the context.

Example:
```
python MetaAgentFive.py --input "Analyze the impact of artificial intelligence on job markets." --context "Recent advancements in machine learning have led to increased automation in various industries."
```

## Output

The script will output the following information:

1. Input Data
2. Analysis (answers to predefined questions)
3. Generated Class Labels
4. Chosen Reasoning Template
5. Filled Template
6. Final Output (including reasoning process, conclusion strength, and suggestions)
7. Final Answer

## Customization

You can modify the `questions` list in the `meta_reasoning_loop` method to change the analytical questions used by the agent.

## License

BSD.


