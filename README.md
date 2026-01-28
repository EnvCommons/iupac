# IUPAC Chemistry Environment

OpenReward environment for bidirectional chemistry nomenclature and structure representation tasks.

## Overview

This environment tests agents on converting between IUPAC chemical nomenclature and SMILES (Simplified Molecular Input Line Entry System) representations. It includes two task types:

1. **iupac2smiles**: Given an IUPAC name, predict the SMILES representation
2. **smiles2iupac**: Given a SMILES string, predict the IUPAC name

## Task Details

### Data Source
All compounds are sourced from the PubChem database via their PUG-REST API.

### Splits
- `iupac2smiles_train`: 1,000 compounds
- `iupac2smiles_test`: 200 compounds
- `smiles2iupac_train`: 1,000 compounds (non-overlapping CIDs)
- `smiles2iupac_test`: 200 compounds (non-overlapping CIDs)

**Total**: 2,400 unique compounds across 4 splits

### Validation Strategy

**SMILES Validation (Deterministic)**
- Uses RDKit for canonicalization
- Handles equivalent representations (e.g., `c1ccccc1` ≡ `C1=CC=CC=C1` for benzene)
- Provides clear error messages for invalid SMILES syntax

**IUPAC Validation (LLM Grader)**
- Uses gpt-5-mini for flexible name matching
- Handles nomenclature variations (e.g., "propan-1-ol" ≡ "1-propanol")
- Accounts for locant positioning and formatting differences

## Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/EnvCommons/iupac.git
cd iupac

# Install dependencies (requires Python 3.11+)
pip install -r requirements.txt

# Run the server
python server.py
```

The server will start on `http://0.0.0.0:8080`.

### Docker

```bash
# Build the image
docker build -t iupac:latest .

# Run the container
docker run -p 8080:8080 iupac:latest
```

**Note**: The Docker image is approximately 1-2 GB due to RDKit dependencies.

## Usage

### Test with Agent

```bash
export OPENAI_API_KEY=your_api_key_here
python test_agent.py
```

This will test both task types (iupac2smiles and smiles2iupac) using the OpenAI Responses API.

### OpenReward Client

```python
from openreward import OpenReward
import asyncio

async def main():
    client = OpenReward(base_url="http://localhost:8080")

    environment = client.environments.get(name="local/IUPAC")
    tasks = await environment.list_tasks(split="iupac2smiles_test")

    # Create session and interact
    async with environment.session(
        task=tasks[0],
        secrets={"openai_api_key": "your_key"}
    ) as session:
        prompt = await session.get_prompt()
        print(prompt[0].text)

        # Submit answer
        result = await session.call_tool(
            "submit_answer",
            {"answer": "CCO"}  # Example SMILES for ethanol
        )
        print(f"Reward: {result.reward}")

asyncio.run(main())
```

## Tool Specification

### `submit_answer`

Submit your final answer for the chemistry task.

**Parameters:**
- `answer` (string, required):
  - For `iupac2smiles` tasks: A valid SMILES string
  - For `smiles2iupac` tasks: An IUPAC chemical name

**Returns:**
- `reward`: 1.0 for correct, 0.0 for incorrect
- `finished`: Always `True` (episode ends after submission)
- `blocks`: Feedback text with correctness and analysis
- `metadata`: Detailed validation results

**Example:**

```json
{
  "answer": "c1ccccc1"
}
```

## Examples

### Example 1: iupac2smiles

**Prompt:**
```
Given the IUPAC name 'ethanol', provide the SMILES representation.
```

**Correct Answers:** `CCO`, `OCC` (both canonicalize to the same structure)

**Feedback:**
```
✅ Correct!

Your answer: CCO
Canonical form: CCO

This is the correct SMILES representation for the given IUPAC name.
```

### Example 2: smiles2iupac

**Prompt:**
```
Given the SMILES 'CC(C)O', provide the IUPAC name.
```

**Correct Answers:** `propan-2-ol`, `2-propanol`, `isopropanol` (grader recognizes equivalence)

**Feedback:**
```
The predicted IUPAC name "propan-2-ol" is chemically equivalent to the reference "2-propanol". Both names refer to the same structure...

✅ Correct

Reference IUPAC name: 2-propanol
Your answer: propan-2-ol
```

## Data

### Regenerating Data

If you need to regenerate the task data from PubChem:

```bash
python download_data.py --output-dir data
```

**Options:**
- `--dry-run`: Test with only 10 compounds per split
- `--output-dir`: Specify output directory (default: `data`)

**Note**: Full data acquisition takes 5-10 minutes due to PubChem rate limits (5 requests/second).

### Data Format

Each JSON file contains tasks with the following structure:

```json
{
  "task_id": "iupac2smiles_train_0001",
  "task_type": "iupac2smiles",
  "cid": 123,
  "split": "iupac2smiles_train",
  "question": "Given the IUPAC name 'ethanol', provide the SMILES representation."
}
```

**Important**: Answer data (`smiles`, `iupac`) is stored separately and never exposed via `list_tasks()`.

## Architecture

- **Base Class**: `Environment` (simple single-turn pattern)
- **No Sandbox**: All validation runs in-process
- **Dependencies**: RDKit (SMILES), OpenAI (IUPAC grading)

## Requirements

- Python 3.11+
- OpenAI API key (for IUPAC grading)
- Dependencies: `rdkit`, `openai`, `openreward`, `pydantic`, `fastapi`, `uvicorn`

## Deployment

This environment is designed for deployment on OpenReward Runtime Service (ORS).

**Namespace**: `EnvCommons/iupac`

See the OpenReward documentation for deployment instructions.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{iupac_openreward,
  title={IUPAC Chemistry Environment for OpenReward},
  author={EnvCommons},
  year={2025},
  url={https://github.com/EnvCommons/iupac}
}
```

## Acknowledgments

- Compound data sourced from [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
- SMILES validation powered by [RDKit](https://www.rdkit.org/)
- IUPAC validation uses OpenAI's gpt-5-mini model
