"""
AI Career Agent — Groq + Serper.dev + Strict PPO filter + Auto model fallback.
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

# ── Auto model fallback list (tried in order) ─────────────────────────────────
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
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
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)
            raw = raw.strip()
            print(f"  ✅ Model {model} responded ({len(raw)} chars)")
            return raw
        except Exception as e:
            last_error = e
            print(f"  ⚠️  Model {model} failed: {e}")
            time.sleep(2)
            continue
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


# ── Serper.dev Search ─────────────────────────────────────────────────────────
def serper_search(query: str, num: int = 6) -> list[dict]:
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPAPI_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num, "gl": "in", "hl": "en"},
            timeout=15,
        )
        response.raise_for_status()
        return [
            {"title": r.get("title",""), "link": r.get("link",""), "snippet": r.get("snippet","")}
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


# ── Step 1: Search ────────────────────────────────────────────────────────────
def search_jobs(profile: dict) -> tuple[list[dict], set]:
    year = date.today().year
    queries = [
        f"AI ML internship Hyderabad {year} apply PPO stipend 2027 batch",
        f"GenAI LLM intern Hyderabad India {year} PPO 2026 2027 batch",
        f"software engineering internship Hyderabad {year} PPO 2027 batch",
        f"data science ML intern Hyderabad {year} full time offer 2027",
        f"remote AI ML internship India {year} PPO stipend 2027 batch",
        f"Google Microsoft Amazon Hyderabad internship {year} 2027 batch",
        f"AI ML internship site:unstop.com {year} 2027 batch",
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
    real_urls   = {r["link"].strip().rstrip("/").lower() for r in all_results if r.get("link")}
    print(f"\n  📦 Raw: {before} → Unique: {len(all_results)} results\n")
    return all_results, real_urls


# ── Format Results ────────────────────────────────────────────────────────────
def format_results(results: list[dict], limit: int = 40) -> str:
    out = ""
    for i, r in enumerate(results[:limit], 1):
        snippet = r["snippet"][:300].replace("\n", " ")
        out    += f"[{i}] {r['title']}\n    URL: {r['link']}\n    {snippet}\n\n"
    return out


# ── Step 2: Extract Jobs ──────────────────────────────────────────────────────
def extract_jobs(profile: dict, raw_results: list[dict], real_urls: set) -> list[dict]:
    skills      = ", ".join(profile.get("skills",     [])[:10])
    frameworks  = ", ".join(profile.get("frameworks", [])[:8])
    grad_year   = profile.get("graduation_year", "2027")
    name        = profile.get("name", "Candidate")
    today       = date.today().strftime("%B %d, %Y")
    search_text = format_results(raw_results)

    prompt = f"""
You are an AI career agent. Today: {today}.
Candidate: {name} | Skills: {skills} | Frameworks: {frameworks} | Grad: {grad_year} (pre-final year, 4th year B.Tech)

LOCATION RULE: Only Hyderabad or Remote/WFH. If not mentioned assume Remote.
BATCH RULE: Include 2027 batch, 2026+2027, or batch not mentioned. Skip "2026 only".
LINK RULE: Copy EXACT URL. Use listing page if no apply URL. "NOT FOUND" only if zero URL.
STIPEND RULE: Only if explicitly in snippet. Otherwise "Not publicly available". Never invent.
EXTRACTION: Be generous. Include anything that looks like an internship. Aim for 10-15 results.

=== SEARCH RESULTS ===
{search_text}
======================

Return ONLY a raw JSON array. No markdown. Start [ end ]

[{{"company":"Name","role":"Title","location":"Hyderabad or Remote","mode":"Remote/Hybrid/Onsite","duration":"X months or Not specified","stipend":"amount or Not publicly available","expected_fte_ctc":"X LPA or Not publicly available","required_skills":["s1","s2"],"apply_link":"EXACT URL or NOT FOUND","deadline":"date or Not specified","date_posted":"date or Not specified","source":"LinkedIn/Unstop/etc"}}]
""".strip()

    print("🤖 Step 2: Extracting jobs via Groq...")
    time.sleep(1)

    raw  = groq_complete([
        {"role": "system", "content": "Return only valid JSON arrays. Start [ end ]. No other text."},
        {"role": "user",   "content": prompt},
    ], max_tokens=4000)

    jobs = safe_parse_json_array(raw)
    if not jobs:
        print("  ⚠️  No jobs extracted.")
        return []

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
        f"{company} internship full time offer experience India 2024 2025",
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
        print(f"  🔍 Verifying: {job.get('company','')}...")
        info = verify_company(job.get("company",""), job.get("role",""))
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
Skills: {skills} | Min FTE CTC: {min_ctc} LPA | Graduation: 2027 (pre-final year, 4th year B.Tech)
Location: Hyderabad or Remote ONLY | Min rating: {MIN_RATING}/5

{summary}

REJECT if ANY: not Hyderabad/Remote, rating below {MIN_RATING}, scam/zero web presence, severe negative reviews, stipend below Rs.{MIN_STIPEND_INR}/month, PPO Low or no evidence, apply_link NOT FOUND or fake, role requires 2026 grad only, role unrelated to AI/ML/SWE/Backend/GenAI.

ACCEPT only if ALL: Hyderabad/Remote, real company, PPO High/Medium WITH evidence, rating {MIN_RATING}+ or unknown funded startup, positive reviews, real apply link, role matches skills.

DATA RULES: Stipend/CTC/Rating from verification data ONLY. Never invent. apply_link copy exactly — if NOT FOUND reject job. ppo_evidence must be specific — if none reject job.

Quality over quantity. Return 0 if nothing passes. No markdown. Start [ end ]

[{{"rank":1,"company":"","role":"","location":"","mode":"","duration":"","stipend":"from data or Not publicly available","expected_fte_ctc":"from data or Not publicly available","ppo_probability":"High or Medium ONLY","ppo_evidence":"specific evidence from reviews","company_rating":"X/5 or Not publicly available","intern_review_summary":"2-3 sentences from REAL reviews","required_skills":[],"why_strong_match":"2 sentences","missing_skills":[],"apply_link":"real URL only","deadline":"","date_posted":"","source":"","verified":true}}]
""".strip()

    print("🤖 Step 4: Scoring and filtering...")
    time.sleep(1)

    raw  = groq_complete([
        {"role": "system", "content": "Strict career advisor. Return only valid JSON arrays. No markdown. Quality over quantity."},
        {"role": "user",   "content": prompt},
    ], max_tokens=5000)

    jobs = safe_parse_json_array(raw)
    if not jobs:
        print("  ⚠️  No jobs passed scoring.")
        return []

    before = len(jobs)
    jobs   = [j for j in jobs if j.get("ppo_probability","").lower() in ("high","medium")]
    print(f"  🎯 PPO filter: {before} → {len(jobs)}")

    before = len(jobs)
    jobs   = [j for j in jobs if is_real_link(j.get("apply_link",""))]
    print(f"  🔗 Link filter: {before} → {len(jobs)}")

    for job in jobs:
        job["apply_link"] = clean_link(job.get("apply_link",""))

    before = len(jobs)
    jobs   = [j for j in jobs if _rating_passes(j.get("company_rating","")) and is_valid_location(j.get("location",""))]
    print(f"  ⭐ Rating + location: {before} → {len(jobs)} passed all filters\n")

    return deduplicate_jobs(jobs)


def _rating_passes(rating_str: str) -> bool:
    if not rating_str or rating_str.lower() in ("not publicly available","n/a",""):
        return True
    m = re.search(r"(\d+\.?\d*)", rating_str)
    return float(m.group(1)) >= MIN_RATING if m else True


# ── Main ──────────────────────────────────────────────────────────────────────
def run_agent() -> list[dict]:
    from memory import filter_new_jobs, remember_jobs, print_memory_stats

    profile = load_resume_profile()
    print("🧠 Checking memory...")
    print_memory_stats()

    raw_results, real_urls = search_jobs(profile)
    raw_jobs               = extract_jobs(profile, raw_results, real_urls)

    if not raw_jobs:
        print("⚠️  No jobs after extraction.")
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