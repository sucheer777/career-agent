"""
AI Career Agent — Groq + Serper.dev + Strict PPO filter + Verified links only.
"""

import os
import json
import re
import time
import requests
from datetime import date
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = (
    os.environ.get("GROK_API_KEY")
    or os.environ.get("GROQ_API_KEY")
)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY environment variable.")
if not SERPAPI_KEY:
    raise ValueError("Set SERPAPI_KEY environment variable.")

RESUME_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "resume_profile.json")
client = Groq(api_key=GROQ_API_KEY)

ALLOWED_LOCATIONS = ["hyderabad", "remote", "work from home", "wfh"]
MIN_RATING        = 3.5
MIN_STIPEND_INR   = 10000

FAKE_LINK_PATTERNS = [
    r"/job[s]?/\d+$",
    r"/careers?/\d+$",
    r"/position[s]?/\d+$",
    r"123456",
    r"000000",
]


# ── Load Profile ──────────────────────────────────────────────────────────────
def load_resume_profile() -> dict:
    with open(RESUME_PROFILE_PATH, "r") as f:
        return json.load(f)


# ── Validators ────────────────────────────────────────────────────────────────
def is_valid_location(location: str) -> bool:
    loc = location.lower().strip()
    return any(a in loc for a in ALLOWED_LOCATIONS)


def is_real_link(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    if "NOT FOUND" in url.upper():
        return False
    if "apply link not found" in url.lower():
        return False
    for p in FAKE_LINK_PATTERNS:
        if re.search(p, url, re.IGNORECASE):
            return False
    return True


def clean_link(url: str) -> str:
    return url if is_real_link(url) else "⚠️ Apply link not found — search on company careers page"


# ── Serper.dev Search ─────────────────────────────────────────────────────────
def serper_search(query: str, num: int = 6) -> list[dict]:
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY":    SERPAPI_KEY,
                "Content-Type": "application/json",
            },
            json={
                "q":   query,
                "num": num,
                "gl":  "in",
                "hl":  "en",
            },
            timeout=10,
        )
        response.raise_for_status()
        data    = response.json()
        results = []
        for r in data.get("organic", []):
            results.append({
                "title":   r.get("title",   ""),
                "link":    r.get("link",    ""),
                "snippet": r.get("snippet", ""),
            })
        return results
    except Exception as e:
        print(f"  ⚠️  Serper error for '{query[:40]}': {e}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate_results(results: list[dict]) -> list[dict]:
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
    seen_keys  = set()
    unique     = []

    for job in jobs:
        link = job.get("apply_link", "").strip().rstrip("/").lower()
        key  = f"{job.get('company','').lower()}::{job.get('role','').lower()}"

        if key in seen_keys:
            print(f"  🚫 Duplicate: {job.get('company')} — {job.get('role')}")
            continue
        if is_real_link(link) and link in seen_links:
            print(f"  🚫 Duplicate link: {job.get('company')} — {job.get('role')}")
            continue

        if is_real_link(link):
            seen_links.add(link)
        seen_keys.add(key)
        unique.append(job)

    return unique


# ── Step 1: Search for Jobs ───────────────────────────────────────────────────
def search_jobs(profile: dict) -> tuple[list[dict], set]:
    year = date.today().year

    queries = [
        f"AI ML internship Hyderabad {year} apply PPO stipend 2027 batch",
        f"GenAI LLM intern Hyderabad India {year} PPO 2026 2027 batch",
        f"software engineering internship Hyderabad {year} PPO 2027 batch eligible",
        f"data science ML intern Hyderabad {year} full time offer 2027 graduating",
        f"remote AI ML internship India {year} PPO stipend 2027 batch",
        f"Google Microsoft Amazon Hyderabad internship {year} 2027 batch",
        f"AI ML internship site:unstop.com {year} 2026 2027 batch",
        f"LLM agentic AI intern remote India {year} PPO 2027",
        f"deep learning intern Hyderabad {year} apply PPO 2027 batch",
        f"MLOps backend AI intern Hyderabad remote {year} 2027 batch",
        f"site:linkedin.com AI ML intern Hyderabad {year} 2027 batch",
        f"site:internshala.com AI ML internship Hyderabad {year} 2027",
        f"penultimate year intern AI ML India {year} Hyderabad remote",
        f"pre final year internship AI ML Hyderabad {year}",
    ]

    all_results = []
    print("🔍 Step 1: Searching via Serper.dev...")

    for query in queries:
        results = serper_search(query, num=6)
        all_results.extend(results)
        print(f"  ✅ '{query[:55]}' → {len(results)} results")
        time.sleep(0.3)

    before      = len(all_results)
    all_results = deduplicate_results(all_results)
    real_urls   = {
        r["link"].strip().rstrip("/").lower()
        for r in all_results if r.get("link")
    }

    print(f"\n  📦 Raw: {before} → Unique: {len(all_results)} results\n")
    return all_results, real_urls


# ── Format Results for Prompt ─────────────────────────────────────────────────
def format_results(results: list[dict], limit: int = 40) -> str:
    out = ""
    for i, r in enumerate(results[:limit], 1):
        snippet = r["snippet"][:350].replace("\n", " ")
        out    += f"[{i}] {r['title']}\n    URL: {r['link']}\n    {snippet}\n\n"
    return out


# ── Step 2: Extract Jobs ──────────────────────────────────────────────────────
def extract_jobs(
    profile: dict,
    raw_results: list[dict],
    real_urls: set,
) -> list[dict]:

    skills     = ", ".join(profile.get("skills",     [])[:10])
    frameworks = ", ".join(profile.get("frameworks", [])[:8])
    grad_year  = profile.get("graduation_year", "2027")
    name       = profile.get("name", "Candidate")
    today      = date.today().strftime("%B %d, %Y")
    search_text= format_results(raw_results)

    prompt = f"""
You are an AI career agent. Today: {today}.
Candidate: {name} | Skills: {skills} | Frameworks: {frameworks} | Grad: {grad_year} (pre-final year student, 4th year B.Tech)

LOCATION RULE: Only extract jobs in Hyderabad or Remote/WFH. Skip everything else.
If location is not mentioned in the snippet, assume Remote and include it.

EXTRACTION RULE:
- Be GENEROUS when extracting — if a result looks like it COULD be an internship, include it
- Do NOT skip results just because they lack full details
- Missing stipend, duration, deadline = still include with "Not publicly available"
- If company name is clear and role looks like internship → include it
- Extract from job boards, company pages, LinkedIn, Unstop, Internshala, Wellfound etc.

LINK RULE:
- apply_link MUST be copied EXACTLY from the URL in search results
- DO NOT invent or modify URLs
- If no direct apply URL exists, use the listing page URL from search results
- Only write "NOT FOUND" if absolutely no URL exists

STIPEND RULE:
- Only write stipend if explicitly mentioned
- Otherwise "Not publicly available"
- NEVER invent numbers

BATCH RULE:
- Include jobs that accept 2027 batch
- Include jobs that accept 2026 OR 2027 batch
- Include jobs that don't mention batch year at all
- SKIP only if job STRICTLY says "2026 batch only" or "already graduated only"

Extract ALL possible internship listings from:

=== SEARCH RESULTS ===
{search_text}
======================

Return ONLY a raw JSON array. No markdown. Start [ end ]
Include as many as possible — aim for 10-15 results minimum.

[{{
  "company": "Name",
  "role": "Title",
  "location": "Hyderabad or Remote",
  "mode": "Remote/Hybrid/Onsite",
  "duration": "X months or Not specified",
  "stipend": "exact amount or Not publicly available",
  "expected_fte_ctc": "X LPA or Not publicly available",
  "required_skills": ["s1", "s2"],
  "apply_link": "EXACT URL from results or NOT FOUND",
  "deadline": "date or Not specified",
  "date_posted": "date or Not specified",
  "source": "LinkedIn/Unstop/Company Site/etc"
}}]
""".strip()

    print("🤖 Step 2: Extracting jobs via Groq...")
    time.sleep(2)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Return only valid JSON arrays. Start [ end ]. No other text."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)

    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)

    jobs = json.loads(raw)

    # Validate links against real search URLs
    for job in jobs:
        link = job.get("apply_link", "")
        norm = link.strip().rstrip("/").lower()
        if not is_real_link(link) or norm not in real_urls:
            job["apply_link"] = "NOT FOUND"

    before = len(jobs)
    jobs   = [j for j in jobs if is_valid_location(j.get("location", ""))]
    print(f"  ✅ Extracted {before} → {len(jobs)} after location filter\n")

    return deduplicate_jobs(jobs)


# ── Step 3: Verify Companies ──────────────────────────────────────────────────
def verify_company(company: str, role: str) -> str:
    queries = [
        f"{company} internship PPO conversion rate return offer India",
        f"{company} intern review ambitionbox glassdoor rating stipend",
        f"{company} internship to full time offer experience India 2024 2025",
    ]

    info = ""
    for query in queries:
        results = serper_search(query, num=3)
        for r in results:
            info += f"{r.get('title','')} — {r.get('snippet','')[:250]}\n"
        time.sleep(0.3)

    return info[:900]


def verify_all_companies(jobs: list[dict]) -> list[dict]:
    print("🔎 Step 3: Verifying companies via Serper.dev...")
    verified = []

    for job in jobs:
        company = job.get("company", "")
        role    = job.get("role",    "")
        print(f"  🔍 Verifying: {company}...")
        info    = verify_company(company, role)
        verified.append({"job": job, "info": info})
        time.sleep(0.3)

    print(f"  ✅ Done verifying {len(verified)} companies.\n")
    return verified


# ── Step 4: Score and Filter ──────────────────────────────────────────────────
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
Verification:
{info}
---"""

    prompt = f"""
You are an extremely strict AI career advisor for {name}. Today: {today}.
Skills: {skills} | Min FTE CTC: {min_ctc} LPA | Graduation: 2027 (pre-final year, currently in 4th year)
Location: Hyderabad or Remote ONLY | Min rating: {MIN_RATING}/5

{summary}

=== STRICT REJECTION RULES ===
REJECT a job if ANY of these are true:
1. Location is NOT Hyderabad or Remote
2. Company rating confirmed below {MIN_RATING}/5
3. Scam, fake, or zero web presence
4. Multiple severe negative intern reviews
5. Stipend confirmed below Rs.{MIN_STIPEND_INR}/month
6. PPO probability is Low
7. NO evidence of PPO/FTE conversion anywhere in reviews or web data
8. apply_link is fake, missing, "NOT FOUND", or cannot be verified
9. Candidate is a pre-final year student (graduating 2027) — eligible for roles
   that accept 2026 OR 2027 batch, or don't specify batch year
10. Role not related to AI/ML/SWE/Backend/GenAI/Data Science
11. Role strictly requires graduation before 2027 with no exceptions

=== STRICT ACCEPTANCE RULES ===
ACCEPT only if ALL of these are true:
1. Hyderabad or Remote location confirmed
2. Company is real with verifiable web presence
3. PPO probability is HIGH or MEDIUM with actual evidence
4. Evidence of PPO/FTE conversion found in reviews or company data
5. Company rating {MIN_RATING}/5 or above (unknown ok ONLY for funded startups)
6. Intern reviews are positive or mixed-positive
7. apply_link is a real working URL from search results
8. Role matches candidate skills in AI/ML/SWE/GenAI/Backend

=== PPO EVIDENCE RULES ===
High PPO: multiple reviews confirm PPO offered, high conversion rate mentioned
Medium PPO: some reviews mention PPO or FTE offer, conversion rate unclear
Low PPO: reviews say no PPO, or company rarely converts → REJECT
Not publicly available + no evidence → REJECT

=== STRICT DATA RULES ===
- Stipend: from verification data ONLY. "Not publicly available" if not found. NEVER invent.
- CTC: from verification data ONLY. "Not publicly available" if not found. NEVER invent.
- Rating: from verification data ONLY. "Not publicly available" if not found.
- apply_link: copy EXACTLY from job data. NEVER generate. If NOT FOUND → REJECT the job.
- intern_review_summary: from REAL reviews in verification data ONLY.
  If no real reviews found → write "No public intern reviews found."
- ppo_evidence: exact quote or clear summary from verification data showing PPO history.
  If no evidence found → write "No PPO evidence found" and REJECT the job.

=== FINAL INSTRUCTION ===
It is BETTER to return 0 jobs than to return unverified or low-quality ones.
Only return jobs you are genuinely confident about.
No markdown. Start [ end ]

Each object MUST have ALL these keys:
{{
  "rank": 1,
  "company": "",
  "role": "",
  "location": "",
  "mode": "",
  "duration": "",
  "stipend": "from data only or Not publicly available",
  "expected_fte_ctc": "from data only or Not publicly available",
  "ppo_probability": "High or Medium ONLY",
  "ppo_evidence": "specific evidence from reviews or company data",
  "company_rating": "X/5 or Not publicly available",
  "intern_review_summary": "2-3 honest sentences from REAL reviews only",
  "required_skills": [],
  "why_strong_match": "2 sentences specific to this candidate's profile",
  "missing_skills": [],
  "apply_link": "real working URL only",
  "deadline": "",
  "date_posted": "",
  "source": "",
  "verified": true
}}
""".strip()

    print("🤖 Step 4: Scoring and filtering (strict PPO + verified links)...")
    time.sleep(2)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Extremely strict career advisor. Return only valid JSON arrays. No markdown. Quality over quantity."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.1,
        max_tokens=5000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)

    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)

    jobs = json.loads(raw)

    # Hard PPO filter
    before = len(jobs)
    jobs   = [
        j for j in jobs
        if j.get("ppo_probability", "").lower() in ("high", "medium")
    ]
    print(f"  🎯 PPO filter: {before} → {len(jobs)} (High/Medium only)")

    # Hard link filter
    before = len(jobs)
    jobs   = [
        j for j in jobs
        if is_real_link(j.get("apply_link", ""))
    ]
    print(f"  🔗 Link filter: {before} → {len(jobs)} (real links only)")

    # Hard location + rating filter
    before = len(jobs)
    jobs   = [
        j for j in jobs
        if _rating_passes(j.get("company_rating", ""))
        and is_valid_location(j.get("location", ""))
    ]
    print(f"  ⭐ Rating + location filter: {before} → {len(jobs)} passed all filters\n")

    return deduplicate_jobs(jobs)


def _rating_passes(rating_str: str) -> bool:
    if not rating_str or rating_str.lower() in ("not publicly available", "n/a", ""):
        return True
    m = re.search(r"(\d+\.?\d*)", rating_str)
    if m:
        return float(m.group(1)) >= MIN_RATING
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def run_agent() -> list[dict]:
    from memory import filter_new_jobs, remember_jobs, print_memory_stats

    profile = load_resume_profile()

    print("🧠 Checking memory...")
    print_memory_stats()

    raw_results, real_urls = search_jobs(profile)
    raw_jobs               = extract_jobs(profile, raw_results, real_urls)

    if not raw_jobs:
        print("⚠️  No jobs after location filter.")
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