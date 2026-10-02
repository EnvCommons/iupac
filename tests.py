"""Offline tests for the smiles2iupac grading path, with a scripted grader client.

OPSIN runs for real, so Java must be on PATH.

Run: uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests.py -q
"""
import asyncio
from types import SimpleNamespace

import pytest

import iupac as mod
from iupac import IUPAC, GradingError, SubmitAnswerInput

TASK = IUPAC.list_tasks("smiles2iupac_train")[0]


class ScriptedClient:
    """Stands in for openai.AsyncClient; returns the scripted replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _env(replies: list[str]) -> tuple[IUPAC, ScriptedClient]:
    env = IUPAC(task_spec=TASK, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(replies)
    return env, env.client


def _submit(env: IUPAC, answer: str):
    return asyncio.run(env.submit_answer(SubmitAnswerInput(answer=answer)))


def test_verdict_graded_and_ends_episode():
    env, client = _env(["Same structure. <answer>CORRECT</answer>"])
    out = _submit(env, "some-name")
    assert out.reward == 1.0 and out.finished is True and client.calls == 1


def test_verdictless_reply_is_resampled():
    env, client = _env(["", "no tags here", "Different. <answer>INCORRECT</answer>"])
    out = _submit(env, "some-name")
    assert out.reward == 0.0 and out.finished is True and client.calls == 3


def test_verdictless_on_every_attempt_raises_without_reference():
    env, client = _env([""] * mod.GRADER_VERDICT_ATTEMPTS)
    with pytest.raises(GradingError) as exc:
        _submit(env, "some-name")
    assert client.calls == mod.GRADER_VERDICT_ATTEMPTS
    assert env.answer_data["iupac"] not in str(exc.value)
    assert env.submitted == 0


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_empty_answer_not_graded(answer):
    env, client = _env(["<answer>CORRECT</answer>"])
    out = _submit(env, answer)
    assert out.finished is False and out.reward == 0.0 and client.calls == 0
    assert env.submitted == 0
    out = _submit(env, "some-name")
    assert out.finished is True and out.reward == 1.0


def _env_for_cid(cid: int, replies: list[str]) -> tuple[IUPAC, ScriptedClient]:
    task = next(t for t in IUPAC.list_tasks("smiles2iupac_train") if t["cid"] == cid)
    env = IUPAC(task_spec=task, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(replies)
    return env, env.client


def _assert_no_reference(env: IUPAC, out, answer: str) -> None:
    # The agent's own answer is echoed back; everything else must not show the reference.
    visible = (out.blocks[0].text + repr(out.metadata)).replace(answer, "")
    assert env.answer_data["iupac"] not in visible
    assert env.answer_data["smiles"] not in visible


# Names for the same structure as the reference, written differently from it.
@pytest.mark.parametrize("cid,answer", [
    # 3,8-dimethyl-5-propan-2-ylazulene-1-sulfonic acid, other azulene numbering direction
    (2275, "1,4-dimethyl-7-(propan-2-yl)azulene-3-sulfonic acid"),
    (2275, "1,4-dimethyl-7-isopropylazulene-3-sulfonic acid"),
    # the same azulene as a von Baeyer bicycle
    (2275, "2,8-dimethyl-5-(propan-2-yl)bicyclo[5.3.0]deca-1,3,5,7,9-pentaene-10-sulfonic acid"),
    # 1-(hydroxymethyl)-7-methoxy-2H-isoquinolin-6-one, oxo prefix instead of the -one suffix
    (1384, "1-(hydroxymethyl)-7-methoxy-6-oxo-2H-isoquinoline"),
    # 2-(1,3-dioxobenzo[de]isoquinolin-2-yl)acetic acid with explicit indicated hydrogen
    (2120, "2-(1,3-dioxo-1H-benzo[de]isoquinolin-2(3H)-yl)acetic acid"),
    # 1-hydroxypyridine-2-thione with added hydrogen spelled out
    (1570, "1-hydroxy-1,2-dihydropyridine-2-thione"),
    # 4-amino-3-(4-chlorophenyl)butanoic acid: a stereodescriptor is ignored
    (2284, "(3R)-4-amino-3-(4-chlorophenyl)butanoic acid"),
    # 8-chloro-6-piperazin-1-ylbenzo[b][1,4]benzoxazepine in dibenzo nomenclature
    (2170, "2-chloro-11-(piperazin-1-yl)dibenzo[b,f][1,4]oxazepine"),
])
def test_equivalent_name_graded_correct_without_llm(cid, answer):
    env, client = _env_for_cid(cid, [])
    out = _submit(env, answer)
    assert out.reward == 1.0 and out.finished is True
    assert client.calls == 0 and out.metadata["graded_by"] == "opsin"
    _assert_no_reference(env, out, answer)


@pytest.mark.parametrize("cid,answer", [
    # positional isomer of the sulfonic acid
    (2275, "3,8-dimethyl-5-propan-2-ylazulene-2-sulfonic acid"),
    # a cumulated, non-azulene bicycle
    (2275, "6,10-dimethyl-3-(propan-2-yl)bicyclo[5.3.0]deca-1,2,4,6,8-pentaene-8-sulfonic acid"),
    # a different ring system
    (2120, "N-(carboxymethyl)-1H-indene-1,7-dicarboximide"),
    # the other tautomer of the given structure: tautomers are compared as written
    (1570, "pyridine-2-thiol 1-oxide"),
])
def test_different_structure_graded_incorrect_without_llm(cid, answer):
    env, client = _env_for_cid(cid, [])
    out = _submit(env, answer)
    assert out.reward == 0.0 and out.finished is True
    assert client.calls == 0 and out.metadata["graded_by"] == "opsin"
    assert env.submitted == 1
    _assert_no_reference(env, out, answer)


@pytest.mark.parametrize("verdict,reward", [("CORRECT", 1.0), ("INCORRECT", 0.0)])
def test_unparseable_name_falls_back_to_llm(verdict, reward):
    # azulene has no position 9, so OPSIN cannot parse this name
    env, client = _env_for_cid(2275, [f"Analysis. <answer>{verdict}</answer>"])
    out = _submit(env, "4,6-dimethyl-9-isopropylazulene-2-sulfonic acid")
    assert out.reward == reward and out.finished is True
    assert client.calls == 1 and out.metadata["graded_by"] == "llm"


def test_reference_name_misread_by_opsin_defers_mismatch_to_llm():
    # OPSIN reads this task's reference name as a structure other than the
    # reference SMILES, so its mismatch on the same name is not trusted.
    env, client = _env_for_cid(2170, ["Same name. <answer>CORRECT</answer>"])
    out = _submit(env, env.answer_data["iupac"])
    assert out.reward == 1.0 and client.calls == 1 and out.metadata["graded_by"] == "llm"


def test_opsin_reads_each_name_as_one_line():
    # Read as two lines, the first name would yield ethanol and shift every later result.
    smiles = asyncio.run(mod.opsin_to_smiles(["ethanol\nfoo", "methanol", "not a chemical"]))
    assert smiles[0] is None and smiles[2] is None
    assert mod.canonical_smiles_without_stereo(smiles[1]) == "CO"


def test_reference_smiles_rdkit_rejects_goes_to_llm():
    # This task's reference SMILES has a hypervalent chlorine that RDKit rejects.
    task = next(t for t in IUPAC.list_tasks("smiles2iupac_test") if t["cid"] == 2373)
    env = IUPAC(task_spec=task, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(["Same structure. <answer>CORRECT</answer>"])
    out = _submit(env, env.answer_data["iupac"])
    assert out.reward == 1.0 and env.client.calls == 1 and out.metadata["graded_by"] == "llm"
