"""
IUPAC Chemistry Environment for OpenReward

Bidirectional chemistry task environment with two task types:
1. iupac2smiles: Given IUPAC name → predict SMILES representation
2. smiles2iupac: Given SMILES → predict IUPAC name

Validation:
- SMILES: RDKit canonical comparison with stereochemistry (ether0 benchmark standard)
- IUPAC: gpt-5-mini grader (flexible matching for nomenclature variations)
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Literal

import openai
from pydantic import BaseModel, Field
from rdkit import Chem

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool, Split


import os

if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent


# Reward for a submission made after the task has already been graded. Negative
# so repeat submissions are actively discouraged, not merely left unscored.
REPEAT_SUBMISSION_PENALTY = -0.1


class GradingError(RuntimeError):
    """Raised when an answer's correctness cannot be established."""

# =============================================================================
# Module-Level Data Loading
# =============================================================================

def load_all_tasks() -> dict[str, list[dict]]:
    """Load all task data from JSON files at module import time."""
    data_dir = ENV_PATH / "data"
    all_tasks = {}

    for split in ["iupac2smiles_train", "iupac2smiles_test",
                  "smiles2iupac_train", "smiles2iupac_test"]:
        json_file = data_dir / f"{split}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                all_tasks[split] = json.load(f)
        else:
            print(f"Warning: {json_file} not found. Split '{split}' will be empty.")
            all_tasks[split] = []

    return all_tasks


# Load all tasks once at module import
ALL_TASKS = load_all_tasks()

# Separate answer storage (backend only - never exposed to agents)
ANSWERS = {
    task["task_id"]: {
        "smiles": task["smiles"],
        "iupac": task["iupac"]
    }
    for split_tasks in ALL_TASKS.values()
    for task in split_tasks
}

print(f"Loaded {len(ANSWERS)} IUPAC tasks across {len(ALL_TASKS)} splits")


# =============================================================================
# Grader Template
# =============================================================================

# Uncapped, a looping thinking trace on the served judge ran for up to 40 min and stalled a shared replica.
GRADER_MAX_TOKENS = 16384

IUPAC_GRADER_TEMPLATE = """You are a chemistry expert evaluating IUPAC nomenclature.

Determine if the predicted IUPAC name is chemically equivalent to the reference.
IUPAC nomenclature has many valid variations:
- "propan-1-ol" vs "1-propanol"
- "dimethyl" vs "di-methyl"
- Locant positioning variations
- Stereochemistry notation differences

Given SMILES: {smiles}
Reference IUPAC: {reference_iupac}
Predicted IUPAC: {predicted_iupac}

Provide 2-3 sentences of analysis comparing the names, then give your verdict
inside <answer></answer> tags containing exactly CORRECT or INCORRECT:
- <answer>CORRECT</answer> if the names refer to the same chemical structure
- <answer>INCORRECT</answer> if they refer to different structures or if the predicted name is invalid
"""


# =============================================================================
# Pydantic Models
# =============================================================================

class IUPACTaskSpec(BaseModel):
    """Task specification for IUPAC environment."""
    task_id: str
    task_type: Literal["iupac2smiles", "smiles2iupac"]
    cid: int
    split: str
    question: str


class SubmitAnswerInput(BaseModel):
    """Input schema for submit_answer tool."""
    answer: str = Field(
        ...,
        description="Your answer: SMILES string for iupac2smiles tasks, or IUPAC name for smiles2iupac tasks"
    )


# =============================================================================
# Main Environment Class
# =============================================================================

class IUPAC(Environment):
    """
    OpenReward environment for bidirectional IUPAC chemistry tasks.

    Task Types:
    - iupac2smiles: Given IUPAC name, predict SMILES representation
    - smiles2iupac: Given SMILES, predict IUPAC name

    Validation:
    - SMILES answers validated using RDKit canonicalization
    - IUPAC answers validated using gpt-5-mini grader
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.validated = IUPACTaskSpec.model_validate(task_spec)

        # Get answer data from backend storage (not exposed to agents)
        if self.validated.task_id not in ANSWERS:
            raise ValueError(f"Task ID {self.validated.task_id} not found in answer storage")
        self.answer_data = ANSWERS[self.validated.task_id]

        # Initialize OpenAI client for IUPAC grading
        # CRITICAL: API key must come from secrets parameter (never env vars)
        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key required for IUPAC grading. "
                "Pass via secrets={'openai_api_key': '...'}"
            )
        # _call_grader retries itself; SDK retries on top would leave more abandoned generations running upstream.
        self.client = openai.AsyncClient(api_key=api_key, max_retries=0)

        # Graded submissions this session. Only the first is rewarded, so the
        # agent cannot turn repeated submissions into a search against the grader.
        self.submitted = 0

    @classmethod
    def list_splits(cls) -> list[Split]:
        """Return all available splits."""
        return [Split(name="iupac2smiles_train", type="train"), Split(name="iupac2smiles_test", type="test"),
                Split(name="smiles2iupac_train", type="train"), Split(name="smiles2iupac_test", type="test")]

    @classmethod
    def list_tasks(cls, split: Split) -> list[JSONObject]:
        """
        Return task specifications for a given split.

        IMPORTANT: This filters out answer data (smiles, iupac) to prevent leakage.
        Only returns task_id, task_type, cid, split, and question.
        """
        if split not in ALL_TASKS:
            raise ValueError(f"Unknown split: {split}. Available: {cls.list_splits()}")

        # Filter out answer fields to prevent leakage
        return [
            {k: v for k, v in task.items() if k not in ["smiles", "iupac"]}
            for task in ALL_TASKS[split]
        ]

    async def get_prompt(self) -> List[TextBlock]:
        """Return the task prompt as a list of TextBlocks."""
        return [TextBlock(text=self.validated.question)]

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final answer for the chemistry task.

        For iupac2smiles tasks: Provide a SMILES string
        For smiles2iupac tasks: Provide an IUPAC name

        This tool validates your answer and returns reward + feedback.
        The episode ends after calling this tool (finished=True), unless a
        SMILES answer cannot be parsed or canonicalized, in which case it is
        not graded and you may resubmit.
        """
        if self.submitted > 0:
            return ToolOutput(
                blocks=[TextBlock(text="An answer has already been submitted for this task. "
                                       "This episode is over: it is not re-graded, and repeat "
                                       "submissions are penalised (reward -0.1).")],
                metadata={"task_id": self.validated.task_id, "already_submitted": True,
                          "submission_count": self.submitted},
                reward=REPEAT_SUBMISSION_PENALTY,
                finished=True,
            )

        if self.validated.task_type == "iupac2smiles":
            result = await self._validate_smiles(params.answer)
        else:  # smiles2iupac
            result = await self._validate_iupac(params.answer)

        # Only a call that actually compared against the reference counts. An
        # unparseable SMILES returns early with an "error" and finished=False and
        # never reached the comparison, so it neither burns the attempt nor ends
        # the episode; a GradingError propagates and never gets here at all.
        if not (result.metadata or {}).get("error"):
            self.submitted += 1
        return result

    # =========================================================================
    # SMILES Validation (Deterministic with RDKit)
    # =========================================================================

    async def _validate_smiles(self, predicted_smiles: str) -> ToolOutput:
        """
        Validate SMILES answer using RDKit canonicalization with stereochemistry.

        This provides deterministic validation by converting both predicted
        and expected SMILES to canonical form (with stereochemistry preserved)
        and comparing them. Follows the ether0 benchmark standard.

        Examples:
        - c1ccccc1 == C1=CC=CC=C1 (benzene - equivalent)
        - CCO == OCC (ethanol - equivalent)
        - Preserves R/S and E/Z stereochemistry for chiral molecules
        """
        # Parse and canonicalize predicted SMILES
        pred_mol = Chem.MolFromSmiles(predicted_smiles.strip())
        # An empty string parses to a molecule with no atoms: not an answer either.
        if pred_mol is None or pred_mol.GetNumAtoms() == 0:
            return ToolOutput(
                blocks=[TextBlock(
                    text="❌ Invalid SMILES format. Your answer could not be parsed as a valid SMILES string. "
                         "It was not graded; submit a corrected SMILES."
                )],
                metadata={
                    "error": "Invalid SMILES",
                    "predicted": predicted_smiles,
                    "task_type": "iupac2smiles"
                },
                reward=0.0,
                finished=False
            )

        # Canonicalize with stereochemistry preservation (following ether0 benchmark)
        try:
            pred_canonical = Chem.MolToSmiles(pred_mol, canonical=True, isomericSmiles=True)
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"❌ Invalid SMILES. Your answer parsed but could not be canonicalized: {str(e)}. "
                         "It was not graded; submit a corrected SMILES."
                )],
                metadata={
                    "error": f"Uncanonicalizable SMILES: {str(e)}",
                    "predicted": predicted_smiles,
                    "task_type": "iupac2smiles"
                },
                reward=0.0,
                finished=False
            )

        # Canonicalize expected SMILES
        expected_mol = Chem.MolFromSmiles(self.answer_data["smiles"])
        if expected_mol is None:
            # Exception messages reach the agent, so the reference is only logged.
            print(f"GRADING ERROR: reference SMILES for cid {self.validated.cid} could not be "
                  f"parsed: {self.answer_data['smiles']!r}")
            raise GradingError(f"Reference SMILES for cid {self.validated.cid} could not be parsed")
        expected_canonical = Chem.MolToSmiles(expected_mol, canonical=True, isomericSmiles=True)

        # Compare canonical forms
        is_correct = (pred_canonical == expected_canonical)
        reward = 1.0 if is_correct else 0.0

        # Generate feedback
        if is_correct:
            feedback = f"""✅ Correct!

Your answer: {predicted_smiles}
Canonical form: {pred_canonical}

This is the correct SMILES representation for the given IUPAC name."""
        else:
            feedback = f"""❌ Incorrect.

Your answer: {predicted_smiles}
Your canonical SMILES: {pred_canonical}

The structures do not match."""

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "predicted": predicted_smiles,
                "predicted_canonical": pred_canonical,
                "correct": is_correct,
                "task_type": "iupac2smiles",
                "cid": self.validated.cid
            },
            reward=reward,
            finished=True
        )

    # =========================================================================
    # IUPAC Validation (LLM Grader)
    # =========================================================================

    async def _validate_iupac(self, predicted_iupac: str) -> ToolOutput:
        """
        Validate IUPAC name using gpt-5-mini grader.

        This provides flexible validation for IUPAC nomenclature variations.
        The LLM grader understands chemical equivalence despite formatting differences.

        Examples:
        - "propan-1-ol" == "1-propanol" (equivalent)
        - "dimethyl" == "di-methyl" (equivalent)
        """
        grader_prompt = IUPAC_GRADER_TEMPLATE.format(
            smiles=self.answer_data["smiles"],
            reference_iupac=self.answer_data["iupac"],
            predicted_iupac=predicted_iupac
        )

        grading_response = await self._call_grader(grader_prompt)

        verdicts = re.findall(
            r"<answer>\s*(INCORRECT|CORRECT)\s*</answer>",
            grading_response,
            re.IGNORECASE
        )
        if not verdicts:
            # Exception messages reach the agent and the grader's response
            # discusses the reference name, so the response is only logged.
            print(f"GRADING ERROR: IUPAC grader returned no verdict: {grading_response!r}")
            raise GradingError("IUPAC grader returned no <answer>CORRECT|INCORRECT</answer> verdict")

        is_correct = verdicts[-1].upper() == "CORRECT"
        reward = 1.0 if is_correct else 0.0

        # Format feedback for agent
        result_label = "✅ Correct" if is_correct else "❌ Incorrect"
        feedback = f"""{result_label}

Your answer: {predicted_iupac}"""

        return ToolOutput(
            blocks=[TextBlock(text=feedback)],
            metadata={
                "predicted": predicted_iupac,
                "correct": is_correct,
                "task_type": "smiles2iupac",
                "cid": self.validated.cid
            },
            reward=reward,
            finished=True
        )

    async def _call_grader(self, grader_prompt: str, max_attempts: int = 4) -> str:
        """Call the grader with exponential backoff, re-raising if it never lands.

        After ``max_attempts`` the last exception propagates so the SDK turns it
        into ToolFailed, rather than the failure being swallowed into a
        fabricated reward.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "user", "content": grader_prompt}],
                    max_completion_tokens=GRADER_MAX_TOKENS,
                    stream=False
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 30)
                    print(
                        f"GRADER API ERROR: gpt-5-mini | {e} | retry in {wait}s "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc
