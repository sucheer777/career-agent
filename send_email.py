"""
Email sender — formats internship results and sends via Resend.
"""

import os
import resend
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
RESEND_API_KEY  = os.environ.get("RESEND_API_KEY")
EMAIL_FROM      = os.environ.get("EMAIL_FROM")   # e.g. jobs@yourdomain.com
EMAIL_TO        = os.environ.get("EMAIL_TO")     # your personal email


# ── Format Jobs as Plain Text ─────────────────────────────────────────────────
def format_jobs_text(jobs: list[dict]) -> str:
    today     = date.today().strftime("%B %d, %Y")
    lines     = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("=" * 60)
    lines.append(f"  AI CAREER AGENT — DAILY INTERNSHIP DIGEST")
    lines.append(f"  {today}")
    lines.append("=" * 60)
    lines.append(f"\nTotal internships found: {len(jobs)}\n")

    # ── Quick Summary ─────────────────────────────────────────────────────────
    if jobs:
        top         = jobs[0]
        high_ppo    = next((j for j in jobs if j.get("ppo_probability") == "High"), None)

        lines.append("── QUICK SUMMARY ──────────────────────────────────────")
        lines.append(f"🥇 Top Pick      : {top['company']} — {top['role']}")

        if high_ppo:
            lines.append(f"🎯 Best PPO Odds : {high_ppo['company']} — {high_ppo['role']}")

        stipends = [
            j for j in jobs
            if j.get("stipend") and j["stipend"] != "Not publicly available"
        ]
        if stipends:
            lines.append(f"💰 Has Stipend   : {len(stipends)} out of {len(jobs)} listings")

        lines.append("")

    # ── Job Listings ──────────────────────────────────────────────────────────
    lines.append("── TOP INTERNSHIPS ────────────────────────────────────")

    for job in jobs:
        rank     = job.get("rank", "?")
        company  = job.get("company", "N/A")
        role     = job.get("role", "N/A")
        location = job.get("location", "N/A")
        mode     = job.get("mode", "N/A")
        duration = job.get("duration", "N/A")
        stipend  = job.get("stipend", "Not publicly available")
        fte_ctc  = job.get("expected_fte_ctc", "Not publicly available")
        ppo      = job.get("ppo_probability", "Not publicly available")
        match    = job.get("why_strong_match", "")
        missing  = job.get("missing_skills", [])
        skills   = ", ".join(job.get("required_skills", []))
        link     = job.get("apply_link", "N/A")
        deadline = job.get("deadline", "Not specified")
        posted   = job.get("date_posted", "Not specified")
        source   = job.get("source", "N/A")

        lines.append(f"\n{'─'*55}")
        lines.append(f"#{rank}  {company.upper()}")
        lines.append(f"    Role        : {role}")
        lines.append(f"    Location    : {location} ({mode})")
        lines.append(f"    Duration    : {duration}")
        lines.append(f"    Stipend     : {stipend}")
        lines.append(f"    FTE CTC     : {fte_ctc}")
        lines.append(f"    PPO Odds    : {ppo}")
        lines.append(f"    Skills      : {skills}")
        lines.append(f"    Why Match   : {match}")

        if missing:
            lines.append(f"    Gap Skills  : {', '.join(missing)}")

        lines.append(f"    Posted      : {posted}")
        lines.append(f"    Deadline    : {deadline}")
        lines.append(f"    Source      : {source}")
        lines.append(f"    Apply Link  : {link}")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines.append(f"\n{'='*60}")
    lines.append("  Apply to ALL relevant roles TODAY.")
    lines.append("  Sent by your AI Career Agent 🤖")
    lines.append(f"{'='*60}\n")

    return "\n".join(lines)


# ── Send Email ────────────────────────────────────────────────────────────────
def send_email(jobs: list[dict]):
    resend.api_key = RESEND_API_KEY

    today   = date.today().strftime("%B %d, %Y")
    subject = f"🤖 Daily Internship Digest — {len(jobs)} Jobs Found | {today}"
    body    = format_jobs_text(jobs)

    print(f"📧 Sending email to {EMAIL_TO}...")

    response = resend.Emails.send({
        "from":    EMAIL_FROM,
        "to":      EMAIL_TO,
        "subject": subject,
        "text":    body,
    })

    print(f"✅ Email sent! ID: {response['id']}")
    return response


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with dummy data
    dummy = [{
        "rank": 1,
        "company": "Test Company",
        "role": "ML Intern",
        "location": "Bangalore",
        "mode": "Hybrid",
        "duration": "6 months",
        "stipend": "₹30,000/month",
        "expected_fte_ctc": "12 LPA",
        "ppo_probability": "High",
        "required_skills": ["Python", "PyTorch"],
        "why_strong_match": "Strong ML project experience matches role perfectly.",
        "missing_skills": [],
        "apply_link": "https://example.com/apply",
        "deadline": "August 15, 2026",
        "date_posted": "July 28, 2026",
        "source": "LinkedIn"
    }]
    send_email(dummy)