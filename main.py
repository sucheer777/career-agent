"""
main.py — Entry point for the AI Career Agent.
GitHub Actions runs this file every day.
"""
from memory import print_memory_stats
import sys
import json
import traceback
from datetime import date

from agent import run_agent
from send_email import send_email



def main():
    today = date.today().strftime("%B %d, %Y")
    print(f"\n{'='*55}")
    print(f"  AI CAREER AGENT STARTING — {today}")
    print(f"{'='*55}\n")

    # ── Step 1: Run Grok Agent ─────────────────────────────
    print("📡 Step 1: Searching for internships via Grok...")
    try:
        jobs = run_agent()
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not jobs:
        print("⚠️  No jobs found today. Exiting.")
        sys.exit(0)

    print(f"\n✅ Found {len(jobs)} internships.\n")

    # ── Step 2: Print Summary to Console (visible in GitHub Actions logs) ──
    print("── TOP 5 PICKS TODAY ──────────────────────────────")
    for job in jobs[:5]:
        print(
            f"  #{job.get('rank')} {job.get('company')} "
            f"— {job.get('role')} "
            f"[{job.get('ppo_probability')} PPO]"
        )
    print("")

    # ── Step 3: Save results to JSON (artifact in GitHub Actions) ──────────
    output_file = f"jobs_{date.today().isoformat()}.json"
    try:
        with open(output_file, "w") as f:
            json.dump(jobs, f, indent=2)
        print(f"💾 Saved results to {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save JSON: {e}")

    # ── Step 4: Send Email ─────────────────────────────────
    print("\n📧 Step 2: Sending daily digest email...")
    try:
        send_email(jobs)
    except Exception as e:
        print(f"❌ Email failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Done ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ✅ ALL DONE — Check your inbox!")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()