"""
AI Career Agent — uses Grok API (with live web search) to find internships.
"""

import os
import json
import re
from datetime import date
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
GROK_API_KEY = os.environ.get("GROK_API_KEY")
RESUME_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "resume_profile.json")

client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)


# ── Load Profile ──────────────────────────────────────────────────────────────
def load_resume_profile() -> dict:
    with open(RESUME_PROFILE_PATH, "r") as f:
        return json.load(f)


# ── Build Prompt ──────────────────────────────────────────────────────────────
def build_search_prompt(profile: dict) -> str:
    skills      = ", ".join(profile.get("skills", []))
    languages   = ", ".join(profile.get("languages", []))
    frameworks  = ", ".join(profile.get("frameworks", []))
    libraries   = ", ".join(profile.get("libraries", []))
    technologies= ", ".join(profile.get("technologies", []))
    roles       = ", ".join(profile.get("preferred_roles", []))
    locations   = ", ".join(profile.get("preferred_locations", []))
    grad_year   = profile.get("graduation_year", "2026")
    degree      = profile.get("degree", "B.Tech CS")
    name        = profile.get("name", "Candidate")
    min_ctc     = profile.get("minimum_fte_ctc_lpa", 8)
    today       = date.today().strftime("%B %d, %Y")

    return f"""
You are an expert AI Career Agent with live internet access. Today is {today}.

Search the internet RIGHT NOW for internship opportunities for this candidate:

=== CANDIDATE PROFILE ===
Name: {name}
Degree: {degree}
Graduation Year: {grad_year}
Skills: {skills}
Languages: {languages}
Frameworks: {frameworks}
Libraries: {libraries}
Technologies: {technologies}
Projects: {json.dumps(profile.get("projects", []), indent=2)}
Experience: {json.dumps(profile.get("experience", []), indent=2)}
Certifications: {", ".join(profile.get("certifications", []))}
Achievements: {", ".join(profile.get("achievements", []))}
Preferred Roles: {roles}
Preferred Locations: {locations}
Minimum FTE CTC: {min_ctc} LPA
=========================

=== SEARCH SOURCES ===
Search ALL of these:
- LinkedIn Jobs
- Google, Microsoft, Amazon, Meta, NVIDIA, Apple, Adobe, Atlassian Careers
- Stripe, Databricks, Snowflake, Salesforce, Uber, Airbnb Careers
- Flipkart, Meesho, Razorpay, PhonePe, Groww, Swiggy, Zomato, CRED, Zepto Careers
- Goldman Sachs, JP Morgan, Visa, Mastercard Tech Careers
- Unstop, Wellfound, AngelList, Internshala (verified only)
- Y Combinator Jobs, HackerEarth Hiring
- Any verified startup career pages

=== FILTERING RULES ===
Only return internships satisfying ALL:
✔ Role is one of: SWE / AI / ML / Deep Learning / Backend / GenAI / Data Science / Applied AI / MLOps / Full Stack AI / AI Research
✔ Currently OPEN — posted within last 30 days
✔ Legitimate company — no spam or scams
✔ PPO / FTE conversion possible OR explicitly stated
✔ Expected FTE CTC at least {min_ctc} LPA (or global equivalent)
✔ Eligible for graduation year {grad_year}

=== RANKING WEIGHTS ===
40% Resume match
20% PPO/FTE probability
15% Stipend/Salary
10% Company reputation
10% Engineering quality
5%  Application deadline

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array of up to 15 internships.
No explanation. No markdown. No code fences. Just raw JSON.

Each object must have EXACTLY these keys:
{{
  "rank": <integer 1-15>,
  "company": "<company name>",
  "role": "<exact role title>",
  "location": "<city, country or Remote>",
  "mode": "<Remote | Hybrid | Onsite>",
  "duration": "<e.g. 2 months, 6 months>",
  "stipend": "<monthly amount with currency or 'Not publicly available'>",
  "expected_fte_ctc": "<e.g. 12 LPA or 'Not publicly available'>",
  "ppo_probability": "<High | Medium | Low | Not publicly available>",
  "required_skills": ["skill1", "skill2"],
  "why_strong_match": "<1-2 sentences explaining fit>",
  "missing_skills": ["skill1"] or [],
  "apply_link": "<direct URL — only real verified links>",
  "deadline": "<date or 'Not specified'>",
  "date_posted": "<date or 'Not specified'>",
  "source": "<e.g. LinkedIn, Company Career Page, Unstop>"
}}

=== STRICT RULES ===
- NEVER invent apply links. Only URLs found via search.
- NEVER invent stipends or PPO rates. Write "Not publicly available".
- SKIP any job that is closed or has a dead link.
- Return ONLY the JSON array. Nothing else.
""".strip()


# ── Run Agent ─────────────────────────────────────────────────────────────────
def run_agent() -> list[dict]:
    profile = load_resume_profile()
    prompt  = build_search_prompt(profile)

    print("🤖 Querying Grok with live web search...")

    response = client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert AI career agent with live internet access. "
                    "Always search the web before answering. "
                    "Return only raw valid JSON arrays with no extra text."
                ),
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    jobs = json.loads(raw)
    print(f"✅ Grok found {len(jobs)} internships.")
    return jobs


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    jobs = run_agent()
    print(json.dumps(jobs, indent=2))