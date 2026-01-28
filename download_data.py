"""
PubChem Data Acquisition Script for IUPAC Environment

Fetches compound data from PubChem PUG-REST API and generates task files for:
- iupac2smiles_train: 1,000 compounds
- iupac2smiles_test: 200 compounds
- smiles2iupac_train: 1,000 compounds (non-overlapping CIDs)
- smiles2iupac_test: 200 compounds (non-overlapping CIDs)
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import requests


# PubChem API configuration
PUBCHEM_API_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IUPACName,IsomericSMILES/JSON"
RATE_LIMIT_DELAY = 0.2  # 5 requests per second = 0.2 seconds between requests
BATCH_SIZE = 100  # Maximum CIDs per request
MAX_IUPAC_LENGTH = 200  # Filter out very long IUPAC names

# Data splits configuration
SPLITS_CONFIG = {
    "iupac2smiles_train": {"count": 1000, "cid_start": 1},
    "iupac2smiles_test": {"count": 200, "cid_start": 1000},
    "smiles2iupac_train": {"count": 1000, "cid_start": 1200},
    "smiles2iupac_test": {"count": 200, "cid_start": 2200},
}

# Fixed random seed for reproducibility
RANDOM_SEED = 42


def fetch_compounds_batch(cid_list: list[int], retry_count: int = 3) -> list[dict[str, Any]]:
    """
    Fetch compound data from PubChem for a batch of CIDs.

    Args:
        cid_list: List of compound IDs to fetch
        retry_count: Number of retries on failure

    Returns:
        List of compound dictionaries with 'CID', 'IUPACName', and 'CanonicalSMILES'
    """
    cid_str = ",".join(map(str, cid_list))
    url = f"{PUBCHEM_API_URL}?cid={cid_str}"

    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                    return data["PropertyTable"]["Properties"]
                else:
                    print(f"Warning: Unexpected response format for CIDs {cid_list[0]}-{cid_list[-1]}")
                    return []
            elif response.status_code == 404:
                print(f"Warning: No data found for CIDs {cid_list[0]}-{cid_list[-1]}")
                return []
            else:
                print(f"Warning: HTTP {response.status_code} for CIDs {cid_list[0]}-{cid_list[-1]}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        except requests.exceptions.RequestException as e:
            print(f"Error fetching CIDs {cid_list[0]}-{cid_list[-1]}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)

    return []


def filter_valid_compounds(compounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Filter compounds to ensure data quality.

    Removes compounds with:
    - Missing IUPAC name or SMILES
    - IUPAC names longer than MAX_IUPAC_LENGTH characters
    - Invalid SMILES (basic check)

    Args:
        compounds: List of compound dictionaries

    Returns:
        Filtered list of valid compounds
    """
    valid = []

    for compound in compounds:
        # Check required fields exist
        if "CID" not in compound or "IUPACName" not in compound or "SMILES" not in compound:
            continue

        iupac = compound["IUPACName"]
        smiles = compound["SMILES"]

        # Filter out missing or overly long IUPAC names
        if not iupac or len(iupac) > MAX_IUPAC_LENGTH:
            continue

        # Filter out missing SMILES
        if not smiles:
            continue

        # Basic SMILES validation (just check it's not empty and has reasonable characters)
        if not smiles.strip() or len(smiles) > 500:
            continue

        valid.append(compound)

    return valid


def collect_compounds(target_count: int, cid_start: int, max_cid: int = 10000) -> list[dict[str, Any]]:
    """
    Collect target number of valid compounds starting from cid_start.

    Args:
        target_count: Number of valid compounds needed
        cid_start: Starting CID (inclusive)
        max_cid: Maximum CID to try (inclusive)

    Returns:
        List of valid compound dictionaries
    """
    valid_compounds = []
    current_cid = cid_start
    batch_count = 0

    print(f"Collecting {target_count} compounds starting from CID {cid_start}...")

    while len(valid_compounds) < target_count and current_cid <= max_cid:
        # Create batch of CIDs
        batch_cids = list(range(current_cid, min(current_cid + BATCH_SIZE, max_cid + 1)))

        # Fetch batch
        print(f"Fetching CIDs {batch_cids[0]}-{batch_cids[-1]} (batch {batch_count + 1})")
        compounds = fetch_compounds_batch(batch_cids)

        # Filter and add valid compounds
        valid = filter_valid_compounds(compounds)
        valid_compounds.extend(valid)

        print(f"  Got {len(valid)} valid compounds (total: {len(valid_compounds)}/{target_count})")

        # Update for next batch
        current_cid += BATCH_SIZE
        batch_count += 1

        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)

        # Checkpoint every 10 batches
        if batch_count % 10 == 0:
            print(f"Checkpoint: Collected {len(valid_compounds)} compounds so far")

    # Trim to exact target count if we got more
    if len(valid_compounds) > target_count:
        print(f"Trimming from {len(valid_compounds)} to {target_count} compounds")
        # Use random sampling with fixed seed for reproducibility
        random.seed(RANDOM_SEED + cid_start)  # Different seed for each split
        valid_compounds = random.sample(valid_compounds, target_count)

    return valid_compounds


def generate_task_file(
    compounds: list[dict[str, Any]],
    split_name: str,
    task_type: str,
    output_dir: Path
) -> None:
    """
    Generate a task JSON file for a given split.

    Args:
        compounds: List of compound dictionaries
        split_name: Name of the split (e.g., "iupac2smiles_train")
        task_type: Type of task ("iupac2smiles" or "smiles2iupac")
        output_dir: Directory to write the JSON file
    """
    tasks = []

    for idx, compound in enumerate(compounds):
        cid = compound["CID"]
        iupac = compound["IUPACName"]
        smiles = compound["SMILES"]

        # Generate question based on task type
        if task_type == "iupac2smiles":
            question = f"Given the IUPAC name '{iupac}', provide the SMILES representation."
        else:  # smiles2iupac
            question = f"Given the SMILES '{smiles}', provide the IUPAC name."

        task = {
            "task_id": f"{split_name}_{idx:04d}",
            "task_type": task_type,
            "cid": cid,
            "split": split_name,
            "iupac": iupac,
            "smiles": smiles,
            "question": question,
        }

        tasks.append(task)

    # Write to file
    output_file = output_dir / f"{split_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(tasks)} tasks to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Download PubChem data for IUPAC environment")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for JSON files (default: data)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test with minimal data (10 compounds per split)"
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PubChem Data Acquisition for IUPAC Environment")
    print("=" * 60)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE: Fetching only 10 compounds per split for testing\n")
        # Override config for dry run
        for split_config in SPLITS_CONFIG.values():
            split_config["count"] = 10

    # Process each split
    for split_name, config in SPLITS_CONFIG.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {split_name}")
        print(f"{'=' * 60}")

        # Determine task type from split name
        task_type = "iupac2smiles" if "iupac2smiles" in split_name else "smiles2iupac"

        # Collect compounds
        compounds = collect_compounds(
            target_count=config["count"],
            cid_start=config["cid_start"]
        )

        if len(compounds) < config["count"]:
            print(f"⚠️  Warning: Only found {len(compounds)} compounds (target: {config['count']})")

        # Generate task file
        generate_task_file(compounds, split_name, task_type, output_dir)

    print(f"\n{'=' * 60}")
    print("✅ Data acquisition complete!")
    print(f"{'=' * 60}")
    print(f"\nGenerated files in {output_dir}:")
    for split_name in SPLITS_CONFIG:
        file_path = output_dir / f"{split_name}.json"
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"  - {file_path.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
