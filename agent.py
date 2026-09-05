"""
AI Career Agent — Groq + Serper.dev
Strict: recent jobs only, verified stipend+package, FTE evidence, 2027 batch, Hyderabad/Remote.
"""

import os
import json
import re
import time
import requests
from datetime import date, timedelta
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

# ── Filters ───────────────────────────────────────────────────────────────────
ALLOWED_LOCATIONS  = ["hyderabad", "remote", "work from home", "wfh"]
MIN_RATING         = 3.5
MIN_STIPEND_INR    = 10000
MIN_FTE_LPA        = 8
MAX_DAYS_OLD       = 2       # only jobs posted within last 2 days

FAKE_LINK_PATTERNS = [
    r"/job[s]?/\d+$",
    r"/careers?/\d+$",
    r"/position[s]?/\d+$",
    r"123456",
    r"000000",
]

# ── Groq model fallback list ──────────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
    "deepseek-r1-distill-llama-70b",
    "llama-3.1-8b-instant",
]

# ── Groq call with auto fallback ──────────────────────────────────────────────
def groq_complete(messages: list, max_tokens: int = 4000) -> str:
    last_error = None
    for model in GROQ_MODELS:
        try:
            print(f"  🤖 Trying model: {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            raw = re.sub(r"<think>.*?</think>",       "", raw, flags=re.DOTALL)
            raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
            raw = re.sub(r"^```(?:json)?\s*",         "", raw)
            raw = re.sub(r"\s*```$",                  "", raw)
            raw = raw.strip()
            print(f"  ✅ Model {model} responded ({len(raw)} chars)")
            return raw
        except Exception as e:
            last_error = e
            print(f"  ⚠️  Model {model} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


# ── Safe JSON parse ───────────────────────────────────────────────────────────
def safe_parse_json_array(raw: str) -> list:
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    if not raw or raw.strip() in ("[]", "[ ]"):
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse failed: {e}")
        print(f"  Raw (first 300): {repr(raw[:300])}")
        return []


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
    return url if is_real_link(url) else "Search on company careers page"


def _rating_passes(rating_str: str) -> bool:
    if not rating_str or rating_str.lower() in ("not publicly available", "n/a", ""):
        return True
    m = re.search(r"(\d+\.?\d*)", rating_str)
    return float(m.group(1)) >= MIN_RATING if m else True


# ── Serper.dev Search ─────────────────────────────────────────────────────────
def serper_search(query: str, num: int = 6, days: int = 2) -> list[dict]:
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPAPI_KEY, "Content-Type": "application/json"},
            json={
                "q":      query,
                "num":    num,
                "gl":     "in",
                "hl":     "en",
                "tbs":    f"qdr:d{days}",   # filter: past N days
            },
            timeout=15,
        )
        response.raise_for_status()
        return [
            {
                "title":   r.get("title",   ""),
                "link":    r.get("link",    ""),
                "snippet": r.get("snippet", ""),
                "date":    r.get("date",    ""),
            }
            for r in response.json().get("organic", [])
        ]
    except Exception as e:
        print(f"  ⚠️  Serper error for '{query[:40]}': {e}")
        return []


def serper_search_no_filter(query: str, num: int = 3) -> list[dict]:
    """Search without date filter — used for company verification."""
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPAPI_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num, "gl": "in", "hl": "en"},
            timeout=15,
        )
        response.raise_for_status()
        return [
            {
                "title":   r.get("title",   ""),
                "link":    r.get("link",    ""),
                "snippet": r.get("snippet", ""),
            }
            for r in response.json().get("organic", [])
        ]
    except Exception as e:
        print(f"  ⚠️  Serper error for '{query[:40]}': {e}")
        return []


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate_results(results: list[dict]) -> list[dict]:
    seen_urls, seen_titles, unique = set(), set(), []
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
    seen_links, seen_keys, unique = set(), set(), []
    for job in jobs:
        link = job.get("apply_link", "").strip().rstrip("/").lower()
        key  = f"{job.get('company','').lower()}::{job.get('role','').lower()}"
        if key in seen_keys:
            print(f"  🚫 Duplicate job: {job.get('company')} — {job.get('role')}")
            continue
        if is_real_link(link) and link in seen_links:
            print(f"  🚫 Duplicate link: {job.get('company')} — {job.get('role')}")
            continue
        if is_real_link(link):
            seen_links.add(link)
        seen_keys.add(key)
        unique.append(job)
    return unique


# ── Step 1: Search ────────────────────────────────────────────────────────────
def search_jobs(profile: dict) -> tuple[list[dict], set]:
    year  = date.today().year
    today = date.today().strftime("%B %d %Y")

    queries = [
        f"AI ML internship Hyderabad {year} 2027 batch stipend PPO apply now",
        f"GenAI LLM intern Hyderabad {year} 2027 batch PPO full time offer",
        f"software engineering internship Hyderabad {year} 2027 batch PPO stipend",
        f"data science ML intern Hyderabad {year} 2027 batch FTE offer stipend",
        f"remote AI ML internship India {year} 2027 batch PPO stipend apply",
        f"deep learning NLP intern Hyderabad remote {year} 2027 batch",
        f"AI ML internship site:unstop.com {year} Hyderabad 2027 batch",
        f"AI ML internship site:internshala.com Hyderabad {year} stipend",
        f"site:linkedin.com/jobs AI ML intern Hyderabad {year} 2027 batch",
        f"Google Microsoft Amazon Hyderabad intern {year} 2027 batch PPO",
        f"Flipkart Meesho Swiggy Zomato intern Hyderabad {year} AI ML",
        f"agentic AI LangGraph LLM intern Hyderabad remote {year} stipend",
        f"MLOps backend AI intern Hyderabad remote {year} 2027 graduating",
        f"pre final year AI ML internship Hyderabad {year} stipend PPO",
    ]

    all_results = []
    print(f"🔍 Step 1: Searching for jobs posted in last {MAX_DAYS_OLD} days...")

    for query in queries:
        results = serper_search(query, num=6, days=MAX_DAYS_OLD)
        all_results.extend(results)
        print(f"  ✅ '{query[:55]}' → {len(results)} results")
        time.sleep(0.3)

    before      = len(all_results)
    all_results = deduplicate_results(all_results)
    real_urls   = {r["link"].strip().rstrip("/").lower() for r in all_results if r.get("link")}

    print(f"\n  📦 Raw: {before} → Unique: {len(all_results)} results (last {MAX_DAYS_OLD} days)\n")
    return all_results, real_urls


# ── Format Results ────────────────────────────────────────────────────────────
def format_results(results: list[dict], limit: int = 25) -> str:
    out = ""
    for i, r in enumerate(results[:limit], 1):
        snippet = r["snippet"][:120].replace("\n", " ")
        date_str = f" | Posted: {r['date']}" if r.get("date") else ""
        out += f"[{i}] {r['title']}{date_str}\n    URL: {r['link']}\n    {snippet}\n\n"
    return out


# ── Step 2: Extract Jobs ──────────────────────────────────────────────────────
def extract_jobs(profile: dict, raw_results: list[dict], real_urls: set) -> list[dict]:
    skills     = ", ".join(profile.get("skills",     [])[:10])
    frameworks = ", ".join(profile.get("frameworks", [])[:8])
    name       = profile.get("name", "Candidate")
    today      = date.today().strftime("%B %d, %Y")
    search_text= format_results(raw_results)

    prompt = f"""
You are an AI career agent helping {name} find internships. Today is {today}.

CANDIDATE PROFILE:
Skills: {skills}
Frameworks: {frameworks}
Graduation: July 2027 (currently in 4th year B.Tech, pre-final year)

YOUR TASK:
Extract internship job listings from the search results below.

=== STRICT EXTRACTION RULES ===

RULE 1 — RECENCY (MOST IMPORTANT):
- ONLY extract jobs posted within the last 1-2 days from today ({today})
- If posting date is not mentioned but the search result looks recent, include it
- If a job was clearly posted weeks or months ago, SKIP IT

RULE 2 — LOCATION:
- ONLY extract jobs in Hyderabad or Remote/Work From Home
- If location is not mentioned in the snippet, assume Remote and include it
- SKIP jobs that are only in other cities (Mumbai, Bangalore, Delhi etc.) unless they also offer remote

RULE 3 — BATCH ELIGIBILITY:
- INCLUDE: jobs for 2027 batch graduates
- INCLUDE: jobs for 2026 OR 2027 batch
- INCLUDE: jobs that don't mention any specific batch year
- SKIP ONLY if job strictly says "2026 batch only" or "must have already graduated"

RULE 4 — ROLE RELEVANCE:
- ONLY extract roles in: AI, ML, Deep Learning, GenAI, LLM, Data Science, Software Engineering, Backend, MLOps, Applied AI, AI Research
- SKIP unrelated roles (HR, Marketing, Finance, Sales, Design etc.)

RULE 5 — LINKS:
- apply_link must be copied EXACTLY from the URL shown in search results
- Do NOT modify, shorten or reconstruct any URL
- If no URL exists for a job, write "NOT FOUND"

RULE 6 — STIPEND:
- Only write stipend if it is EXPLICITLY mentioned with a number in the snippet
- If not found, write exactly: "Not publicly available"
- NEVER invent or guess a number

=== SEARCH RESULTS (last {MAX_DAYS_OLD} days) ===
{search_text}
===================================================

Return ONLY a raw JSON array. No markdown. No explanation. Start with [ and end with ]
Extract as many valid listings as possible (aim for 10+).

[{{
  "company": "Exact company name",
  "role": "Exact role title from listing",
  "location": "Hyderabad or Remote or Hyderabad/Remote",
  "mode": "Remote or Hybrid or Onsite",
  "duration": "X months or Not specified",
  "stipend": "₹XX,000/month or Not publicly available",
  "expected_fte_ctc": "X LPA or Not publicly available",
  "required_skills": ["skill1", "skill2"],
  "apply_link": "EXACT URL copied from search results or NOT FOUND",
  "deadline": "date or Not specified",
  "date_posted": "date from result or Not specified",
  "source": "LinkedIn / Unstop / Internshala / Company Site / etc"
}}]
""".strip()

    print("🤖 Step 2: Extracting jobs via Groq...")
    time.sleep(1)

    raw  = groq_complete([
        {"role": "system", "content": "You extract job listings from search results. Return only valid JSON arrays. Start with [ and end with ]. No other text whatsoever."},
        {"role": "user",   "content": prompt},
    ], max_tokens=2000)

    jobs = safe_parse_json_array(raw)
    if not jobs:
        print("  ⚠️  No jobs extracted.")
        return []

    # Validate every link
    for job in jobs:
        link = job.get("apply_link", "")
        norm = link.strip().rstrip("/").lower()
        if not is_real_link(link) or norm not in real_urls:
            job["apply_link"] = "NOT FOUND"

    # Hard location filter
    before = len(jobs)
    jobs   = [j for j in jobs if is_valid_location(j.get("location", ""))]
    print(f"  ✅ Extracted {before} → {len(jobs)} after location filter\n")

    return deduplicate_jobs(jobs)


# ── Step 3: Verify Companies ──────────────────────────────────────────────────
def verify_company(company: str, role: str) -> str:
    queries = [
        f"{company} internship PPO return offer FTE conversion rate India",
        f"{company} intern stipend salary package ambitionbox glassdoor",
        f"{company} fresher package LPA salary India 2024 2025",
        f"{company} intern review experience India good bad",
    ]
    info = ""
    for query in queries:
        results = serper_search_no_filter(query, num=3)
        for r in results:
            info += f"{r.get('title','')} — {r.get('snippet','')[:250]}\n"
        time.sleep(0.3)
    return info[:1000]


def verify_all_companies(jobs: list[dict]) -> list[dict]:
    print("🔎 Step 3: Verifying companies (PPO, stipend, package, reviews)...")
    verified = []
    for job in jobs:
        print(f"  🔍 Verifying: {job.get('company','')}...")
        info = verify_company(job.get("company",""), job.get("role",""))
        verified.append({"job": job, "info": info})
        time.sleep(0.3)
    print(f"  ✅ Done verifying {len(verified)} companies.\n")
    return verified


# ── Step 4: Score and Filter ──────────────────────────────────────────────────
def score_and_filter(verified_jobs: list[dict], profile: dict) -> list[dict]:
    skills  = ", ".join(profile.get("skills", [])[:10])
    name    = profile.get("name", "Candidate")
    today   = date.today().strftime("%B %d, %Y")

    summary = ""
    for i, v in enumerate(verified_jobs, 1):
        job  = v["job"]
        info = v["info"]
        summary += f"""
--- JOB {i} ---
Company   : {job.get('company')}
Role      : {job.get('role')}
Location  : {job.get('location')} ({job.get('mode')})
Duration  : {job.get('duration')}
Stipend   : {job.get('stipend')}
FTE CTC   : {job.get('expected_fte_ctc')}
Apply Link: {job.get('apply_link')}
Skills    : {', '.join(job.get('required_skills', []))}
Posted    : {job.get('date_posted')}
Deadline  : {job.get('deadline')}
Source    : {job.get('source')}

WEB VERIFICATION DATA (reviews, PPO, salary, rating):
{info}
--------------
"""

    prompt = f"""
You are an extremely strict AI career advisor. Today is {today}.
You are helping {name}, a pre-final year B.Tech student graduating in July 2027.
Candidate skills: {skills}

Below are internship listings with web verification data about each company.
Your job is to STRICTLY filter and rank only the best opportunities.

=== MANDATORY REJECTION CRITERIA ===
IMMEDIATELY REJECT a job if ANY of the following are true:

1. STIPEND UNKNOWN: No stipend information found anywhere (listing + verification data).
   → We do NOT want free internships. If stipend is completely unknown, REJECT.
   → Exception: If company is a top-tier brand (Google, Microsoft, Amazon, Meta, etc.) where stipend is well-known to be high, you may keep it.

2. FTE PACKAGE UNKNOWN: No fresher/FTE package information found anywhere.
   → If expected FTE CTC for freshers cannot be estimated at ≥{MIN_FTE_LPA} LPA, REJECT.
   → Exception: Same top-tier brand exception applies.

3. NO PPO EVIDENCE: No evidence of PPO or FTE conversion found in reviews or verification data.
   → "Not publicly available" for PPO with no other evidence = REJECT.
   → Must have at least some review or data confirming the company converts interns.

4. BAD REVIEWS: Company has predominantly negative intern reviews (below {MIN_RATING}/5 rating, or multiple reviews mentioning poor experience, no learning, fake PPO promises).

5. FAKE/SCAM COMPANY: Company has no web presence, no reviews, no funding info, no LinkedIn page.

6. WRONG LOCATION: Job is not in Hyderabad or Remote. No exceptions.

7. WRONG BATCH: Job strictly requires 2026 or earlier graduation. Must accept 2027 batch.

8. WRONG ROLE: Not related to AI/ML/GenAI/LLM/Data Science/SWE/Backend/MLOps.

9. FAKE APPLY LINK: apply_link is "NOT FOUND" or does not look like a real job listing URL.

10. LOW STIPEND: Stipend confirmed below ₹{MIN_STIPEND_INR:,}/month. Reject unpaid/underpaid internships.

=== MANDATORY ACCEPTANCE CRITERIA ===
ONLY accept if ALL of the following are true:

1. STIPEND CONFIRMED or strongly implied: ₹{MIN_STIPEND_INR:,}+/month
   (or top-tier company where high stipend is industry-known)

2. FTE PACKAGE: Evidence of ≥{MIN_FTE_LPA} LPA fresher package at this company
   (from reviews, ambitionbox, glassdoor, or company reputation)

3. PPO EVIDENCE: Real evidence that this company converts interns to FTE
   (reviews, LinkedIn posts, intern experiences, company policy)

4. COMPANY QUALITY: Real company with web presence, funding, or brand recognition
   AND rating ≥{MIN_RATING}/5 on Glassdoor/Ambitionbox (or no rating for new funded startups)

5. REVIEWS EXIST: At least some intern review data exists (can be mixed, not strictly positive)

6. LOCATION: Hyderabad or Remote confirmed

7. BATCH: Accepts 2027 batch graduates

8. REAL APPLY LINK: Valid URL to the actual job listing

9. ROLE MATCH: Role matches candidate's AI/ML/LLM/Backend/Data Science skills

=== SCORING (rank by total score) ===
- PPO probability High = +40 points
- PPO probability Medium = +20 points
- Stipend ≥ ₹30,000 = +20 points, ≥ ₹20,000 = +10 points
- FTE CTC ≥ 15 LPA = +20 points, ≥ 10 LPA = +10 points
- Company rating ≥ 4.0 = +15 points, ≥ 3.5 = +10 points
- Top brand (Google/Microsoft/Amazon/Flipkart etc.) = +15 points
- Skills match ≥ 80% = +10 points

=== OUTPUT FORMAT ===
Return ONLY a raw JSON array ranked by score (highest first).
No markdown. No explanation. Start with [ and end with ]
Return empty array [] if nothing passes all criteria.

[{{
  "rank": 1,
  "company": "Company name",
  "role": "Exact role title",
  "location": "Hyderabad or Remote",
  "mode": "Remote or Hybrid or Onsite",
  "duration": "X months or Not specified",
  "stipend": "Verified ₹XX,000/month or Not publicly available",
  "expected_fte_ctc": "X LPA based on verification data or Not publicly available",
  "ppo_probability": "High or Medium (ONLY these two values — never Low or Unknown)",
  "ppo_evidence": "Exact quote or specific summary from reviews proving PPO/FTE conversion",
  "company_rating": "X.X/5 from Glassdoor/Ambitionbox or Not publicly available",
  "fresher_package_evidence": "What the company pays freshers — from reviews/ambitionbox",
  "intern_review_summary": "2-3 honest sentences summarizing real intern reviews found",
  "required_skills": ["skill1", "skill2"],
  "why_strong_match": "2 specific sentences explaining why this matches {name}'s exact profile",
  "missing_skills": ["skill if any"],
  "apply_link": "Real working URL — copied exactly from job data",
  "deadline": "Application deadline or Not specified",
  "date_posted": "When this job was posted",
  "source": "Where the listing was found",
  "verified": true
}}]
""".strip()

    print("🤖 Step 4: Scoring and filtering (strict PPO + stipend + package + reviews)...")
    time.sleep(1)

    raw  = groq_complete([
        {"role": "system", "content": "You are an extremely strict career advisor. Return only valid JSON arrays. No markdown. No explanation. Quality over quantity — return 0 jobs rather than bad ones."},
        {"role": "user",   "content": prompt},
    ], max_tokens=6000)

    jobs = safe_parse_json_array(raw)
    if not jobs:
        print("  ⚠️  No jobs passed scoring.")
        return []

    # Hard PPO filter
    before = len(jobs)
    jobs   = [j for j in jobs if j.get("ppo_probability","").lower() in ("high","medium")]
    print(f"  🎯 PPO filter       : {before} → {len(jobs)}")

    # Hard link filter
    before = len(jobs)
    jobs   = [j for j in jobs if is_real_link(j.get("apply_link",""))]
    print(f"  🔗 Link filter      : {before} → {len(jobs)}")

    # Clean links
    for job in jobs:
        job["apply_link"] = clean_link(job.get("apply_link",""))

    # Rating + location filter
    before = len(jobs)
    jobs   = [
        j for j in jobs
        if _rating_passes(j.get("company_rating",""))
        and is_valid_location(j.get("location",""))
    ]
    print(f"  ⭐ Rating+location  : {before} → {len(jobs)} passed all filters\n")

    return deduplicate_jobs(jobs)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_agent() -> list[dict]:
    from memory import filter_new_jobs, remember_jobs, print_memory_stats

    profile = load_resume_profile()
    today   = date.today().strftime("%B %d, %Y")

    print(f"\n📅 Date: {today}")
    print(f"📍 Location: Hyderabad + Remote only")
    print(f"🎓 Batch: 2027 (pre-final year)")
    print(f"💰 Min stipend: ₹{MIN_STIPEND_INR:,}/month")
    print(f"📦 Min FTE CTC: {MIN_FTE_LPA} LPA")
    print(f"🗓  Jobs posted: last {MAX_DAYS_OLD} days only\n")

    print("🧠 Checking memory for previously seen jobs...")
    print_memory_stats()

    raw_results, real_urls = search_jobs(profile)

    if not raw_results:
        print("⚠️  No search results found.")
        return []

    # Process in batches to stay under Groq token limit
    raw_jobs = []
    batch_size = 25
    for i in range(0, len(raw_results), batch_size):
        batch = raw_results[i:i+batch_size]
        print(f"  📦 Processing batch {i//batch_size + 1}...")
        batch_jobs = extract_jobs(profile, batch, real_urls)
        raw_jobs.extend(batch_jobs)
        time.sleep(3)

    raw_jobs = deduplicate_jobs(raw_jobs)
    print(f"  ✅ Total extracted after all batches: {len(raw_jobs)}\n")

    if not raw_jobs:
        print("⚠️  No jobs extracted.")
        return []

    print("🧠 Filtering previously seen jobs...")
    raw_jobs = filter_new_jobs(raw_jobs)

    if not raw_jobs:
        print("⚠️  All jobs already seen before. Nothing new today.")
        return []

    verified   = verify_all_companies(raw_jobs)
    final_jobs = score_and_filter(verified, profile)

    print("🧠 Final memory check...")
    final_jobs = filter_new_jobs(final_jobs)

    if final_jobs:
        print("💾 Saving new jobs to memory...")
        remember_jobs(final_jobs)
    else:
        print("⚠️  No jobs passed all filters today.")

    return final_jobs


if __name__ == "__main__":
    jobs = run_agent()
    print(json.dumps(jobs, indent=2))
