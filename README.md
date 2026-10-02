# IUPACNames

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/IUPACNames)

## Description

IUPACNames is an environment for evaluating bidirectional chemistry nomenclature and structure representation tasks. It contains 2,400 compounds with tasks for converting between IUPAC chemical names and SMILES (Simplified Molecular Input Line Entry System) representations, sourced from PubChem.

## Capabilities

- IUPAC nomenclature to SMILES conversion
- SMILES to IUPAC nomenclature conversion
- Chemical structure understanding
- Equivalent representation recognition

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are four splits in this environment:

- **iupac2smiles_train**: 1,000 tasks
- **iupac2smiles_test**: 200 tasks
- **smiles2iupac_train**: 1,000 tasks
- **smiles2iupac_test**: 200 tasks

All compounds are sourced from PubChem with non-overlapping CIDs across task types.

## Reward Structure

This is a single-turn environment with two validation strategies:

**SMILES Validation (Deterministic)**: Uses RDKit for canonicalization, handling equivalent representations (e.g., `c1ccccc1` equals `C1=CC=CC=C1` for benzene).

**IUPAC Validation (Structure Match)**: The predicted name is parsed to a structure with [OPSIN](https://github.com/dan2097/opsin) and compared with the reference by RDKit canonical SMILES, ignoring stereochemistry. Charges and tautomers are compared as written, as in SMILES validation. Any name for the right structure is accepted (e.g., "propan-1-ol" equals "1-propanol"; von Baeyer and fused-ring names of the same ring system are equal). Names OPSIN cannot parse are judged by a gpt-5-mini grader, as are non-matching names on the few tasks whose reference name OPSIN reads as a different structure.

Reward is binary: 1.0 if correct, 0.0 if incorrect. For iupac2smiles, a SMILES that is empty or cannot be parsed or canonicalized is not graded: it returns reward 0.0 and the episode stays open so the agent can resubmit.

## Data

Data consists of JSON files containing compounds sourced from PubChem via their PUG-REST API. Each task includes a question, compound identifier, and answer data. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `submit_answer` | Submit SMILES string or IUPAC name. Ends the episode once an answer is graded. |

## Time Horizon

Single-turn. The agent reads the chemistry question and submits one answer.

## Environment Difficulty

[Put environment difficulty here]

## Other Environment Requirements

Java (for OPSIN; installed in the Docker image). OpenAI API key required for IUPAC name grading of names OPSIN cannot parse. Pass via `secrets={"openai_api_key": "..."}`.

## Safety

Agents in IUPACNames convert between chemical representations in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@software{iupacnames2025,
  title={IUPACNames: Chemistry Nomenclature Environment for OpenReward},
  author={{General Reasoning Inc. Team}},
  year={2025},
  url={https://www.openreward.ai/GeneralReasoning/IUPACNames}
}
```
