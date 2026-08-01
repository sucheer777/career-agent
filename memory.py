"""
memory.py — Persistent job memory. Tracks every job ever shown.
Never recommends the same job twice across days.
"""

import os
import json
from datetime import date

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "seen_jobs.json")


# ── Load Memory ───────────────────────────────────────────────────────────────
def load_memory() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {"seen_links": [], "seen_keys": [], "history": []}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


# ── Save Memory ───────────────────────────────────────────────────────────────
def save_memory(memory: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
    print(f"  💾 Memory saved — {len(memory['seen_keys'])} total jobs remembered.")


# ── Filter Out Already Seen Jobs ──────────────────────────────────────────────
def filter_new_jobs(jobs: list[dict]) -> list[dict]:
    memory     = load_memory()
    seen_links = set(memory.get("seen_links", []))
    seen_keys  = set(memory.get("seen_keys", []))

    new_jobs   = []
    skipped    = 0

    for job in jobs:
        link    = job.get("apply_link", "").strip().rstrip("/").lower()
        company = job.get("company", "").strip().lower()
        role    = job.get("role",    "").strip().lower()
        key     = f"{company}::{role}"

        if link in seen_links or key in seen_keys:
            print(f"  🧠 Already seen (skipping): {job.get('company')} — {job.get('role')}")
            skipped += 1
            continue

        new_jobs.append(job)

    print(f"  ✅ {len(new_jobs)} new jobs | {skipped} already seen (filtered out)\n")
    return new_jobs


# ── Save New Jobs to Memory ───────────────────────────────────────────────────
def remember_jobs(jobs: list[dict]):
    memory     = load_memory()
    seen_links = set(memory.get("seen_links", []))
    seen_keys  = set(memory.get("seen_keys", []))
    history    = memory.get("history", [])
    today      = date.today().isoformat()

    for job in jobs:
        link    = job.get("apply_link", "").strip().rstrip("/").lower()
        company = job.get("company", "").strip().lower()
        role    = job.get("role",    "").strip().lower()
        key     = f"{company}::{role}"

        seen_links.add(link)
        seen_keys.add(key)

        # Save to history log
        history.append({
            "date":    today,
            "company": job.get("company"),
            "role":    job.get("role"),
            "link":    job.get("apply_link"),
            "stipend": job.get("stipend"),
            "ppo":     job.get("ppo_probability"),
        })

    memory["seen_links"] = list(seen_links)
    memory["seen_keys"]  = list(seen_keys)
    memory["history"]    = history

    save_memory(memory)


# ── Show Memory Stats ─────────────────────────────────────────────────────────
def print_memory_stats():
    memory = load_memory()
    print(f"  🧠 Memory: {len(memory['seen_keys'])} jobs remembered across all days")
    print(f"  📅 History entries: {len(memory['history'])}\n")