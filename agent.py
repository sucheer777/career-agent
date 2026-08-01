"""
AI Career Agent — Groq + DDGS + location filter + stipend extraction + rating filter.
"""

import os
import json
import re
import time
from datetime import date
from groq import Groq
# NEW
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = (
    os.environ.get("GROK_API_KEY")
    or os.environ.get("GROQ_API_KEY")
)

if not GROQ_API_KEY:
    raise ValueError("No API key found. Set GROQ_API_KEY environment variable.")

RESUME_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "resume_profile.json")
client = Groq(api_key=GROQ_API_KEY)

# ── Location & Quality Config ─────────────────────────────────────────────────
ALLOWED_LOCATIONS  = ["hyderabad", "remote", "work from home", "wfh"]
MIN_RATING         = 3.5   # reject anything below this
MIN_STIPEND_INR    = 10000 # reject if stipend confirmed below this


# ── Load Profile ──────────────────────────────────────────────────────────────
def load_resume_profile() -> dict:
    with open(RESUME_PROFILE_PATH, "r") as f:
        return json.load(f)


# ── Location Filter ───────────────────────────────────────────────────────────
def is_valid_location(location: str) -> bool:
    loc = location.lower().strip()
    return any(allowed in loc for allowed in ALLOWED_LOCATIONS)


# ── Deduplicate ───────────────────────────────────────────────────────────────
def deduplicate_search_results(results: list[dict]) -> list[dict]:
    seen_urls   = set()
    seen_titles = set()
    unique      = []

    for r in results:
        url   = r.get("link",  "").strip().rstrip("/").lower()
        title = r.get("title", "").strip().lower()

        if url in seen_urls or title in seen_titles:
            continue

        seen_urls.add(url)
        seen_titles.add(title)
        unique.append(r)

    return unique


def deduplicate_jobs(jobs: list[dict]) -> list[dict]:
    seen_links = set()
    seen_keys  = set()  # ← must be set, use .add() not assignment
    unique     = []

    for job in jobs:
        link    = job.get("apply_link", "").strip().rstrip("/").lower()
        company = job.get("company",    "").strip().lower()
        role    = job.get("role",       "").strip().lower()
        key     = f"{company}::{role}"

        if link in seen_links or key in seen_keys:
            print(f"  🚫 Duplicate: {job.get('company')} — {job.get('role')}")
            continue

        seen_links.add(link)
        seen_keys.add(key)   # ← .add() not assignment
        unique.append(job)

    return unique


# ── Step 1: Search ────────────────────────────────────────────────────────────
def search_jobs(profile: dict) -> str:
    year = date.today().year

    queries = [
        f"AI ML internship Hyderabad {year} apply stipend",
        f"GenAI LLM intern Hyderabad India {year}",
        f"software engineering internship Hyderabad {year} PPO",
        f"data science ML intern Hyderabad {year} opening",
        f"remote AI ML internship India {year} PPO stipend",
        f"Google Microsoft Amazon Hyderabad internship {year}",
        f"AI ML intern site:unstop.com Hyderabad {year}",
        f"LLM agentic AI intern remote India {year}",
    ]

    all_results = []
    print("🔍 Step 1: Searching (Hyderabad + Remote only)...")

    ddgs = DDGS()
    for query in queries:
        try:
            results = list(ddgs.text(query, max_results=4))
            for r in results:
                all_results.append({
                    "title":   r.get("title", ""),
                    "link":    r.get("href",  ""),
                    "snippet": r.get("body",  ""),
                })
            print(f"  ✅ '{query[:52]}' → {len(results)} results")
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️  Failed: {query[:52]} ({e})")

    before      = len(all_results)
    all_results = deduplicate_search_results(all_results)
    print(f"\n  📦 Raw: {before} → Unique: {len(all_results)}\n")

    formatted = ""
    for i, r in enumerate(all_results[:35], 1):
        snippet = r["snippet"][:150].replace("\n", " ")
        formatted += f"[{i}] {r['title']}\n    URL: {r['link']}\n    {snippet}\n\n"

    return formatted


# ── Step 2: Extract Jobs ──────────────────────────────────────────────────────
def extract_jobs(profile: dict, search_results: str) -> list[dict]:
    skills     = ", ".join(profile.get("skills",     [])[:10])
    frameworks = ", ".join(profile.get("frameworks", [])[:8])
    grad_year  = profile.get("graduation_year", "2027")
    name       = profile.get("name", "Candidate")
    today      = date.today().strftime("%B %d, %Y")

    prompt = f"""
You are an AI career agent. Today: {today}.
Candidate: {name} | Skills: {skills} | Frameworks: {frameworks} | Grad: {grad_year}

LOCATION RULE: Only extract jobs in Hyderabad or Remote/Work From Home. Ignore all others.

Extract internship listings from search results below.
Return ONLY a JSON array starting with [ and ending with ].

=== RESULTS ===
{search_results}
===============

JSON format:
[{{"company":"Name","role":"Title","location":"Hyderabad or Remote","mode":"Remote/Hybrid/Onsite","duration":"X months","stipend":"amount or Not publicly available","expected_fte_ctc":"X LPA or Not publicly available","required_skills":["s1"],"apply_link":"https://url","deadline":"date or Not specified","date_posted":"date or Not specified","source":"LinkedIn etc"}}]

Rules:
- ONLY Hyderabad or Remote locations
- Only URLs from search results
- No duplicates
- Return ONLY the JSON array
""".strip()

    print("🤖 Step 2: Extracting jobs via Groq...")
    time.sleep(2)  # avoid TPM limit

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON arrays. Start with [ end with ]. No other text."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    jobs = json.loads(raw)

    # Hard location filter
    before = len(jobs)
    jobs   = [j for j in jobs if is_valid_location(j.get("location", ""))]
    print(f"  ✅ Extracted {before} → {len(jobs)} after location filter (Hyderabad/Remote only)\n")

    jobs = deduplicate_jobs(jobs)
    return jobs


# ── Step 3: Verify Company ────────────────────────────────────────────────────
def verify_company(company: str, role: str) -> str:
    queries = [
        f"{company} internship review ambitionbox glassdoor rating",
        f"{company} intern stipend PPO conversion India 2025 2026",
    ]

    ddgs = DDGS()
    info = ""

    for query in queries:
        try:
            results = list(ddgs.text(query, max_results=2))
            for r in results:
                info += f"{r.get('title','')} — {r.get('body','')[:250]}\n"
            time.sleep(0.3)
        except Exception:
            continue

    return info[:700]


def verify_all_companies(jobs: list[dict]) -> list[dict]:
    print("🔎 Step 3: Verifying companies (stipend, PPO, reviews, rating)...")
    verified = []

    for job in jobs:
        company = job.get("company", "")
        role    = job.get("role",    "")
        print(f"  🔍 Verifying: {company}...")
        info    = verify_company(company, role)
        verified.append({"job": job, "info": info})
        time.sleep(0.5)

    print(f"  ✅ Done verifying {len(verified)} companies.\n")
    return verified


# ── Step 4: Score, Filter, Extract Stipend ───────────────────────────────────
def score_and_filter(verified_jobs: list[dict], profile: dict) -> list[dict]:
    skills  = ", ".join(profile.get("skills", [])[:10])
    min_ctc = profile.get("minimum_fte_ctc_lpa", 8)
    name    = profile.get("name", "Candidate")
    today   = date.today().strftime("%B %d, %Y")

    summary = ""
    for i, v in enumerate(verified_jobs, 1):
        job  = v["job"]
        info = v["info"]
        summary += f"""
[{i}] {job.get('company')} | {job.get('role')} | {job.get('location')} | {job.get('mode')}
Stipend: {job.get('stipend')} | FTE: {job.get('expected_fte_ctc')}
Link: {job.get('apply_link')}
Skills: {', '.join(job.get('required_skills', []))}
Verification Data:
{info}
---"""

    prompt = f"""
You are a strict AI career advisor for {name}. Today: {today}.
Skills: {skills} | Min FTE CTC: {min_ctc} LPA
Location allowed: Hyderabad or Remote ONLY.
Minimum company rating: {MIN_RATING}/5
Minimum stipend: ₹{MIN_STIPEND_INR:,}/month

Here are internships with web verification data:
{summary}

YOUR RULES:

REJECT if ANY of these:
- Location is NOT Hyderabad or Remote
- Company rating below {MIN_RATING}/5 (from Glassdoor/Ambitionbox)
- Scam, fake, or unknown company with zero web presence
- Multiple very negative intern reviews
- Stipend confirmed below ₹{MIN_STIPEND_INR:,}/month
- No PPO history and very bad reputation

ACCEPT if ALL of these:
- Hyderabad or Remote location
- Rating {MIN_RATING}/5 or above (or funded startup with no rating yet)
- Real company with web presence
- Decent intern reviews
- Stipend ≥ ₹{MIN_STIPEND_INR:,}/month or not publicly confirmed

STIPEND RULE:
- If stipend says "Not publicly available", search the verification data for any mention of stipend/salary
- Extract it if found in reviews
- If still not found, write "Not publicly available"
- NEVER invent a number

RATING RULE:
- Extract exact rating from verification data if mentioned (e.g. "3.8/5", "4.1 out of 5")
- If not found, write "Not publicly available"
- If rating is below {MIN_RATING}, REJECT the company

Return ONLY a JSON array ranked best to worst. No markdown. Start [ end ].

Each object MUST have:
{{"rank":1,"company":"","role":"","location":"Hyderabad or Remote","mode":"","duration":"","stipend":"verified amount or Not publicly available","expected_fte_ctc":"X LPA or Not publicly available","ppo_probability":"High/Medium/Low/Not publicly available","company_rating":"X/5 or Not publicly available","intern_review_summary":"2-3 honest sentences from reviews","required_skills":[],"why_strong_match":"2 sentences","missing_skills":[],"apply_link":"","deadline":"","date_posted":"","source":"","verified":true}}

Return ONLY the JSON array.
""".strip()

    print("🤖 Step 4: Scoring, filtering, extracting stipend from reviews...")
    time.sleep(2)  # avoid TPM limit

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Strict career advisor. Return only valid JSON arrays. No markdown. No extra text."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=5000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    jobs = json.loads(raw)

    # Hard filter: remove low rated jobs that slipped through
    before = len(jobs)
    jobs = [
        j for j in jobs
        if _rating_passes(j.get("company_rating", ""))
        and is_valid_location(j.get("location", ""))
    ]
    print(f"  ✅ {before} → {len(jobs)} passed all filters (location + rating + quality)\n")

    jobs = deduplicate_jobs(jobs)
    return jobs


def _rating_passes(rating_str: str) -> bool:
    """Returns True if rating is above MIN_RATING or not available (give benefit of doubt)."""
    if not rating_str or rating_str.lower() in ("not publicly available", "n/a", ""):
        return True  # unknown rating — let it through, reviewed by Groq already
    match = re.search(r"(\d+\.?\d*)", rating_str)
    if match:
        return float(match.group(1)) >= MIN_RATING
    return True


# ── Main Entry ────────────────────────────────────────────────────────────────
def run_agent() -> list[dict]:
    from memory import filter_new_jobs, remember_jobs, print_memory_stats

    profile = load_resume_profile()

    print("🧠 Checking memory for previously seen jobs...")
    print_memory_stats()

    search_results = search_jobs(profile)
    raw_jobs       = extract_jobs(profile, search_results)

    if not raw_jobs:
        print("⚠️  No new jobs found after location filter.")
        return []

    print("🧠 Filtering previously seen jobs...")
    raw_jobs = filter_new_jobs(raw_jobs)

    if not raw_jobs:
        print("⚠️  All jobs already seen. Nothing new today.")
        return []

    verified   = verify_all_companies(raw_jobs)
    final_jobs = score_and_filter(verified, profile)

    print("🧠 Final memory check...")
    final_jobs = filter_new_jobs(final_jobs)

    if final_jobs:
        print("💾 Saving to memory...")
        remember_jobs(final_jobs)

    return final_jobs


if __name__ == "__main__":
    jobs = run_agent()
    print(json.dumps(jobs, indent=2))