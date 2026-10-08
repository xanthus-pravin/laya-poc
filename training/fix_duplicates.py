"""Remove duplicated records and append 10 genuinely new ones."""
from __future__ import annotations

import json

QUESTIONS_BLOCK = {
    "burnout_risk": {
        "type": "noul",
        "instructions": "Does the conversation contain explicit signs of employee burnout, exhaustion, or being overwhelmed?",
        "criteria": {"true": "team members express exhaustion", "false": "team members sound comfortable"},
    },
    "root_cause": {
        "type": "choice",
        "instructions": "What is the primary negative driver expressed by the team in this chat?",
        "criteria": {
            "overload_and_deadlines": "overload and deadlines",
            "lack_of_support_and_isolation": "lack of support and isolation",
            "none_healthy": "none healthy",
        },
    },
    "psychological_safety": {
        "type": "score",
        "instructions": "Rate the psychological safety of the team based on their communication.",
        "criteria": ["Low: Afraid to speak up", "Medium: Frustrated but vocal", "High: Healthy collaboration"],
    },
}

PATH = "training/dataset.jsonl"

with open(PATH, "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

# Remove the last 10 records (52-61, which were duplicates of 22-31)
rows = rows[:-10]

# 10 new genuinely different records
new_records = [
    {
        "text": "Bhavana Shetty: The client health scoring feature flag went out without my final approval and I only found out when support started escalations.\nRavi Ashwin: That should never happen. Who owns the approval chain?\nBhavana Shetty: It was supposed to be me, but the PM overrode it.\nRavi Ashwin: I will raise the governance gap with the lead. You should not have to find out from tickets.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Mohan Rangan: Seven days straight on the message schema refactor and I am barely functional by evening.\nPadmini Krish: Are you tracking your hours?\nMohan Rangan: I stopped. The lead sees the commit count and that is apparently all that matters.\nPadmini Krish: Then I am not saying anything either. It does not help.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Keshav Nayak: The drift detection window logic keeps changing and nobody can tell me the current spec.\nLeela Bose: Ask in the architecture channel.\nKeshav Nayak: I did. Three days ago. No answer.\nLeela Bose: Then we both guess and hope we are right.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Sonal Kapoor: I have taken three on-call pages in one week and the rotation spreadsheet still shows me for next week.\nDinesh Rawat: Did you flag it?\nSonal Kapoor: Three times. The last response was maybe next quarter.\nDinesh Rawat: I will cover your shift this week. You need the break.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Irfan Qureshi: The token refresh retry loop is consuming half the daily sync and it is wearing everyone down.\nNalini Das: It is a known issue with a documented workaround. Why are we still treating it as new?\nIrfan Qureshi: Because nobody is closing the loop on the permanent fix.\nNalini Das: Fair. Let me draft an RFC and take ownership.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Tanay Mukherjee: My second pull request in a row got approved with no comments, which means nobody read it.\nRitu Saxena: That is how they get merged. I have stopped writing detailed descriptions.\nTanay Mukherjee: So now we just ship blind?\nRitu Saxena: Pretty much. You get used to it.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Geeta Menon: I am genuinely proud of what we shipped on the Cliq migration this week.\nSameer Kulkarni: The backfill ran clean and the metrics look great.\nGeeta Menon: Credit to the whole team, especially the testers.\nSameer Kulkarni: Hear that from leadership too. You earned it.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Pranav Joshi: The RBAC permission matrix we built last month caught three misconfigurations in staging this week.\nEkta Rathi: That is exactly why we invested the time.\nPranav Joshi: Good to see it paying off. Let us share the pattern with the platform team.\nEkta Rathi: Already drafted the runbook. Making it public next Monday.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Meera Shenoy: I want to flag something I saw in the client health scoring output.\nHarsh Vardhan: Go ahead, that is what this channel is for.\nMeera Shenoy: The drift thresholds are triggering too early on seasonal accounts.\nHarsh Vardhan: Good catch. Let us pull the raw data and verify together.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Nitin Bansal: The quarterly retro actions are actually getting closed this time.\nPreeti Malhotra: Short lists and named owners make a real difference.\nNitin Bansal: I am going to propose we keep the cap at three items per retro.\nPreeti Malhotra: Seconded. Quality over quantity.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
]

rows.extend(new_records)

with open(PATH, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Total records now: {len(rows)}")
