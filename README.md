# TabXEval: A Comprehensive Framework for Evaluating Table Extraction Models

This repository contains the code and resources for the TabXEval framework, a comprehensive evaluation framework for table extraction models. The framework provides tools for evaluating and comparing different table extraction approaches, with a focus on accuracy, robustness, and real-world applicability.

## Repository Structure

```
.
├── evaluation_pipeline/     # Core evaluation scripts and utilities
│   ├── eval.py             # Main evaluation script
│   ├── eval_gemini.py      # Gemini model evaluation
│   ├── eval_llama.py       # LLaMA model evaluation
│   ├── fuzzy_table_matching.py  # Fuzzy matching utilities
│   └── comparison_utils.py # Comparison utilities
├── tabxbench/             # Benchmark datasets and tools
├── EVALUATION_OF_MODELS/  # Evaluation results and analysis
└── TabXEval.pdf          # Research paper
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tabxeval.git
cd tabxeval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Running Evaluations

To evaluate a model using the framework:

```bash
python evaluation_pipeline/eval.py \
    --align_prompt path/to/align_prompt.txt \
    --compare_prompt path/to/compare_prompt.txt \
    --input_tables path/to/input_tables.json \
    --output_path path/to/output/
```

### Available Models

The framework supports evaluation of multiple models:
- GPT-4
- Gemini
- LLaMA

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{tabxeval2024,
  title={TabXEval: A Comprehensive Framework for Evaluating Table Extraction Models},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
