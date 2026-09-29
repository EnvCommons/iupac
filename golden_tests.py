"""
Golden Tests for IUPAC Environment

Tests that submitting ground truth answers yields reward = 1.0
Tests both task types:
- iupac2smiles: Submit ground truth SMILES
- smiles2iupac: Submit ground truth IUPAC name
"""

import asyncio
import json
import os
from pathlib import Path

from openreward import AsyncOpenReward


async def load_ground_truth(split: str) -> list[dict]:
    """Load ground truth data from JSON files."""
    data_dir = Path(__file__).parent / "data"
    json_file = data_dir / f"{split}.json"

    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


async def test_task(environment, task, ground_truth, task_type: str, api_key: str):
    """Test a single task by submitting ground truth answer."""
    # Extract task_id from task object
    task_id = task.task_spec.get("task_id") if hasattr(task, 'task_spec') else str(task)

    async with environment.session(task=task, secrets={"OPENAI_API_KEY": api_key}) as session:
        prompt = await session.get_prompt()

        # Extract ground truth answer based on task type
        if task_type == "iupac2smiles":
            answer = ground_truth["smiles"]
            answer_type = "SMILES"
        else:  # smiles2iupac
            answer = ground_truth["iupac"]
            answer_type = "IUPAC"

        # Submit the ground truth answer
        tool_result = await session.call_tool(
            "submit_answer",
            {"answer": answer}
        )

        reward = tool_result.reward
        result_text = tool_result.blocks[0].text if tool_result.blocks else ""

        # Assert reward is 1.0
        assert reward == 1.0, (
            f"Expected reward 1.0 for ground truth answer, got {reward}\n"
            f"Task: {task_id}\n"
            f"Answer: {answer}\n"
            f"Result: {result_text}"
        )

        return {
            "task_id": task_id,
            "task_type": task_type,
            "answer": answer,
            "reward": reward,
            "passed": True
        }


async def test_invalid_then_gold(environment, task, ground_truth, api_key: str):
    """An unparseable SMILES is not graded and keeps the episode open; the gold answer after it is graded."""
    task_id = task.task_spec.get("task_id") if hasattr(task, 'task_spec') else str(task)

    async with environment.session(task=task, secrets={"OPENAI_API_KEY": api_key}) as session:
        await session.get_prompt()

        invalid_result = await session.call_tool("submit_answer", {"answer": "C1CC(("})
        assert invalid_result.reward == 0.0 and not invalid_result.finished, (
            f"Expected an ungraded, unfinished result for invalid SMILES, got "
            f"reward={invalid_result.reward} finished={invalid_result.finished}\n"
            f"Task: {task_id}"
        )

        answer = ground_truth["smiles"]
        tool_result = await session.call_tool("submit_answer", {"answer": answer})
        assert tool_result.reward == 1.0 and tool_result.finished, (
            f"Expected reward 1.0 and finished for ground truth after invalid SMILES, got "
            f"reward={tool_result.reward} finished={tool_result.finished}\n"
            f"Task: {task_id}"
        )

        return {
            "task_id": task_id,
            "task_type": "iupac2smiles",
            "answer": answer,
            "reward": tool_result.reward,
            "passed": True
        }


async def main() -> None:
    """Run golden tests on first 5 examples of each task type."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set - skipping smiles2iupac tests (require LLM grader)")
        print("   iupac2smiles tests will still run (use deterministic RDKit validation)\n")

    or_client = AsyncOpenReward()
    environment = or_client.environments.get(
        name="local/IUPAC",
        base_url="http://localhost:8080"
    )

    results = []

    # Test configurations: (split, task_type, num_tests)
    test_configs = [
        ("iupac2smiles_test", "iupac2smiles", 5),
        ("smiles2iupac_test", "smiles2iupac", 5),
    ]

    for split, task_type, num_tests in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing {split} ({num_tests} examples)")
        print(f"{'='*70}\n")

        # Skip smiles2iupac if no API key
        if task_type == "smiles2iupac" and not api_key:
            print(f"⚠️  Skipping {split} - requires OPENAI_API_KEY for LLM grader\n")
            continue

        # Load ground truth data
        ground_truth_data = await load_ground_truth(split)

        # Get tasks from environment (filtered - no answers)
        tasks = await environment.list_tasks(split=split)

        # Test first N tasks
        for i in range(min(num_tests, len(tasks))):
            task = tasks[i]
            ground_truth = ground_truth_data[i]

            # Extract task_id from task object
            task_id = task.task_spec.get("task_id") if hasattr(task, 'task_spec') else f"{split}_{i}"

            print(f"Test {i+1}/{num_tests}: {task_id}")

            try:
                result = await test_task(
                    environment,
                    task,
                    ground_truth,
                    task_type,
                    api_key or ""
                )
                results.append(result)
                print(f"  ✅ PASSED - Reward: {result['reward']:.1f}")
                print(f"     Answer: {result['answer'][:80]}...")

            except AssertionError as e:
                print(f"  ❌ FAILED")
                print(f"     {e}")
                results.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "passed": False,
                    "error": str(e)
                })
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                results.append({
                    "task_id": task_id,
                    "task_type": task_type,
                    "passed": False,
                    "error": str(e)
                })

    print(f"\n{'='*70}")
    print("Testing invalid SMILES then ground truth (iupac2smiles_test[0])")
    print(f"{'='*70}\n")
    split = "iupac2smiles_test"
    ground_truth_data = await load_ground_truth(split)
    tasks = await environment.list_tasks(split=split)
    task_id = tasks[0].task_spec.get("task_id") if hasattr(tasks[0], 'task_spec') else f"{split}_0"
    try:
        result = await test_invalid_then_gold(environment, tasks[0], ground_truth_data[0], api_key or "")
        results.append(result)
        print(f"  ✅ PASSED - Reward: {result['reward']:.1f}")
    except AssertionError as e:
        print(f"  ❌ FAILED")
        print(f"     {e}")
        results.append({"task_id": task_id, "task_type": "iupac2smiles", "passed": False, "error": str(e)})
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append({"task_id": task_id, "task_type": "iupac2smiles", "passed": False, "error": str(e)})

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("\n🎉 All golden tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        failed_tasks = [r for r in results if not r.get("passed", False)]
        for task in failed_tasks:
            print(f"  - {task['task_id']}: {task.get('error', 'Unknown error')}")

    # Exit with appropriate code
    exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
