"""Offline tests for the smiles2iupac grading path, with a scripted grader client.

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
