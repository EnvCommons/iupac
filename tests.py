"""Offline tests for the smiles2iupac grading path, with a scripted grader client.

OPSIN runs for real, so Java must be on PATH.

Run: uv run --no-project --with-requirements requirements.txt --with pytest python -m pytest tests.py -q
"""
import asyncio
import re
from types import SimpleNamespace

import httpx
import openai
import pytest

import iupac as mod
from iupac import IUPAC, GradingError, SubmitAnswerInput

TASK = IUPAC.list_tasks("smiles2iupac_train")[0]


# A name OPSIN cannot parse; it has a locant, so it reaches the LLM grader.
UNPARSEABLE_NAME = "1-some-name"


class ScriptedClient:
    """Stands in for openai.AsyncClient; returns the scripted replies in order.

    A scripted exception is raised instead of replying. Each call's keyword
    arguments are recorded in ``requests``.
    """

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        self.requests.append(kwargs)
        content = self.replies.pop(0)
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _timeout() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))


def _env(replies: list[str]) -> tuple[IUPAC, ScriptedClient]:
    env = IUPAC(task_spec=TASK, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient(replies)
    return env, env.client


def _submit(env: IUPAC, answer: str):
    return asyncio.run(env.submit_answer(SubmitAnswerInput(answer=answer)))


def test_verdict_graded_and_ends_episode():
    env, client = _env(["Same structure. <answer>CORRECT</answer>"])
    out = _submit(env, UNPARSEABLE_NAME)
    assert out.reward == 1.0 and out.finished is True and client.calls == 1


def test_verdictless_reply_is_resampled_with_low_effort():
    env, client = _env(["no tags here", "Different. <answer>INCORRECT</answer>"])
    out = _submit(env, UNPARSEABLE_NAME)
    assert out.reward == 0.0 and out.finished is True and client.calls == 2
    first, last = client.requests
    assert first["max_completion_tokens"] == mod.GRADER_MAX_TOKENS and "reasoning_effort" not in first
    assert last["reasoning_effort"] == "low" and last["max_completion_tokens"] < mod.GRADER_MAX_TOKENS


def test_grader_requests_are_time_bounded():
    # A tool call that runs past about 10 minutes is lost, so the grader's
    # timeouts must add up to well under that.
    env, client = _env(["", ""])
    _submit(env, UNPARSEABLE_NAME)
    assert all(0 < r["timeout"] for r in client.requests)
    assert sum(r["timeout"] for r in client.requests) <= 450


def test_grader_timeout_moves_to_next_attempt():
    env, client = _env([_timeout(), "Same structure. <answer>CORRECT</answer>"])
    out = _submit(env, UNPARSEABLE_NAME)
    assert out.reward == 1.0 and out.finished is True and client.calls == 2
    assert client.requests[1]["reasoning_effort"] == "low"


def test_verdictless_on_every_attempt_is_not_graded_without_reference():
    env, client = _env(["", _timeout()])
    out = _submit(env, UNPARSEABLE_NAME)
    assert out.finished is False and out.reward == 0.0
    assert client.calls == len(mod.GRADER_ATTEMPTS)
    assert out.metadata["error"] and env.submitted == 0
    _assert_no_reference(env, out, UNPARSEABLE_NAME)
    # The attempt was not used: a resubmission is graded normally.
    client.replies = ["Same structure. <answer>CORRECT</answer>"]
    out = _submit(env, UNPARSEABLE_NAME)
    assert out.reward == 1.0 and out.finished is True and env.submitted == 1


def test_grader_api_failure_still_raises(monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
    env, client = _env([openai.APIConnectionError(request=httpx.Request("POST", "https://x"))] * 4)
    with pytest.raises(openai.APIConnectionError):
        _submit(env, UNPARSEABLE_NAME)
    assert client.calls == 4 and env.submitted == 0


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_empty_answer_not_graded(answer):
    env, client = _env(["<answer>CORRECT</answer>"])
    out = _submit(env, answer)
    assert out.finished is False and out.reward == 0.0 and client.calls == 0
    assert env.submitted == 0
    out = _submit(env, UNPARSEABLE_NAME)
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


def _synthetic_env(monkeypatch, smiles: str, reference_name: str) -> tuple[IUPAC, ScriptedClient]:
    """An environment on a task outside the dataset, with the given reference."""
    task_id = "smiles2iupac_synthetic"
    monkeypatch.setitem(mod.ANSWERS, task_id, {"smiles": smiles, "iupac": reference_name})
    task = {"task_id": task_id, "task_type": "smiles2iupac", "cid": 0,
            "split": "smiles2iupac_test", "question": f"Name {smiles}"}
    env = IUPAC(task_spec=task, secrets={"openai_api_key": "test"})
    env.client = ScriptedClient([])
    return env, env.client


def _assert_rule_rejected(env: IUPAC, client: ScriptedClient, out, answer: str, rule: str) -> None:
    assert out.reward == 0.0 and out.finished is True and env.submitted == 1
    assert client.calls == 0
    assert out.metadata["graded_by"] == "rule" and out.metadata["rule"] == rule
    assert "Not a systematic IUPAC name" in out.blocks[0].text
    _assert_no_reference(env, out, answer)


def _full_opsin_structure(name: str) -> str | None:
    return mod.canonical_smiles_without_stereo(asyncio.run(mod.opsin_to_smiles([name]))[0])


# Trivial names OPSIN resolves to the reference structure, with PubChem's
# SMILES and IUPAC name for it. Without the rules they would grade 1.0.
TRIVIAL_NAME_TASKS = [
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "1,3,7-trimethylpurine-2,6-dione"),
    ("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "2-[4-(2-methylpropyl)phenyl]propanoic acid"),
    ("paracetamol", "CC(=O)NC1=CC=C(C=C1)O", "N-(4-hydroxyphenyl)acetamide"),
    ("morphine", "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",
     "(4R,4aR,7S,7aR,12bS)-3-methyl-2,4,4a,7,7a,13-hexahydro-1H-4,12-methanobenzofuro[3,2-e]isoquinoline-7,9-diol"),
    ("adenosine", "C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O)N",
     "(2R,3R,4S,5R)-2-(6-aminopurin-9-yl)-5-(hydroxymethyl)oxolane-3,4-diol"),
]


@pytest.mark.parametrize("answer,smiles,reference_name", TRIVIAL_NAME_TASKS)
def test_trivial_name_graded_incorrect_without_opsin_or_llm(monkeypatch, answer, smiles, reference_name):
    assert _full_opsin_structure(answer) == mod.canonical_smiles_without_stereo(smiles)
    env, client = _synthetic_env(monkeypatch, smiles, reference_name)
    out = _submit(env, answer)
    _assert_rule_rejected(env, client, out, answer, "missing_locants")


@pytest.mark.parametrize("answer", [name for name, _, _ in TRIVIAL_NAME_TASKS])
def test_systematic_opsin_does_not_resolve_trivial_names(answer):
    full, systematic = (asyncio.run(mod.opsin_to_smiles([answer], systematic_only=only))[0]
                        for only in (False, True))
    assert full is not None and systematic is None


# Names with locants built on a trivial parent, each the reference structure.
@pytest.mark.parametrize("cid,answer", [
    (2153, "1,3-dimethylxanthine"),
    (2193, "androst-4-ene-3,17-dione"),
    (1254, "p-menthan-3-ol"),
    (1868, "6-methoxy-1,2,3,4-tetrahydro-beta-carboline"),
    (1867, "9-beta-D-ribofuranosyl-6-mercaptopurine"),
])
def test_semi_trivial_name_graded_incorrect_without_llm(cid, answer):
    env, client = _env_for_cid(cid, [])
    assert _full_opsin_structure(answer) == mod.canonical_smiles_without_stereo(env.answer_data["smiles"])
    out = _submit(env, answer)
    _assert_rule_rejected(env, client, out, answer, "trivial_name")


@pytest.mark.parametrize("answer,smiles,reference_name", [
    ("5-fluorouracil", "C1=C(C(=O)NC(=O)N1)F", "5-fluoro-1H-pyrimidine-2,4-dione"),
    ("2'-deoxyadenosine", "C1[C@@H]([C@H](O[C@H]1N2C=NC3=C(N=CN=C32)N)CO)O",
     "(2R,3S,5R)-5-(6-aminopurin-9-yl)-2-(hydroxymethyl)oxolan-3-ol"),
])
def test_semi_trivial_name_on_synthetic_task_graded_incorrect(monkeypatch, answer, smiles, reference_name):
    assert _full_opsin_structure(answer) == mod.canonical_smiles_without_stereo(smiles)
    env, client = _synthetic_env(monkeypatch, smiles, reference_name)
    out = _submit(env, answer)
    _assert_rule_rejected(env, client, out, answer, "trivial_name")
    # The systematic reference name still grades correct.
    env, client = _synthetic_env(monkeypatch, smiles, reference_name)
    out = _submit(env, reference_name)
    assert out.reward == 1.0 and out.metadata["graded_by"] == "opsin" and client.calls == 0


@pytest.mark.parametrize("answer", [
    # o/m/p letter locants instead of numerals: OPSIN reads it as the reference
    "N-acetyl-p-aminophenol",
    # a trivial name OPSIN cannot parse, so it would otherwise go to the LLM grader
    "acetaminophen",
])
def test_name_without_locants_graded_incorrect_without_llm(answer):
    env, client = _env_for_cid(1983, ["Same structure. <answer>CORRECT</answer>"])
    out = _submit(env, answer)
    _assert_rule_rejected(env, client, out, answer, "missing_locants")


DIGITLESS_REFERENCE_TASKS = [t for split in ("smiles2iupac_train", "smiles2iupac_test")
                             for t in mod.ALL_TASKS[split] if not re.search(r"[0-9]", t["iupac"])]


def test_digitless_reference_count():
    assert len(DIGITLESS_REFERENCE_TASKS) == 11


@pytest.mark.parametrize("task", DIGITLESS_REFERENCE_TASKS, ids=lambda t: t["iupac"])
def test_digitless_reference_name_graded_correct(task):
    spec = {k: v for k, v in task.items() if k not in ("smiles", "iupac")}
    env = IUPAC(task_spec=spec, secrets={"openai_api_key": "test"})
    env.client = client = ScriptedClient([])
    out = _submit(env, task["iupac"])
    assert out.reward == 1.0 and out.metadata["graded_by"] == "opsin" and client.calls == 0


def test_every_reference_name_graded_correct(monkeypatch):
    """Submitting the reference name scores 1.0 on all smiles2iupac tasks.

    The few names OPSIN cannot compare go to the LLM grader, scripted as CORRECT;
    no reference name may be rejected by the locant or trivial-name rule.
    """
    tasks = [t for split in ("smiles2iupac_train", "smiles2iupac_test") for t in IUPAC.list_tasks(split)]
    assert len(tasks) == 1200

    async def grade(task):
        env = IUPAC(task_spec=task, secrets={"openai_api_key": "test"})
        env.client = ScriptedClient(["Same structure. <answer>CORRECT</answer>"])
        return await env.submit_answer(SubmitAnswerInput(answer=env.answer_data["iupac"]))

    async def grade_all():
        # The module's semaphore may already be bound to another test's event loop.
        monkeypatch.setattr(mod, "_OPSIN_SLOTS", asyncio.Semaphore(4))
        return await asyncio.gather(*(grade(t) for t in tasks))

    results = asyncio.run(grade_all())
    graded_by = [r.metadata["graded_by"] for r in results]
    assert graded_by.count("rule") == 0
    assert [t["cid"] for t, r in zip(tasks, results) if r.reward != 1.0] == []
    assert graded_by.count("llm") <= 7


def test_trivial_name_tokens_file_is_up_to_date(monkeypatch, tmp_path):
    import build_trivial_name_tokens as build
    monkeypatch.setattr(build, "OUTPUT", tmp_path / "trivial_name_tokens.tsv")
    build.main()
    assert build.OUTPUT.read_text(encoding="utf-8") == mod.TRIVIAL_NAME_TOKENS.read_text(encoding="utf-8")
