"""Append 30 new synthetic training records to training/dataset.jsonl.

10 severe (burnout=true, ps=0)
10 moderate (ps=1, mixed root_cause)
10 healthy (burnout=false, ps=2, root_cause=none_healthy)
"""
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

records = [
    # ── SEVERE (10) ──────────────────────────────────────────────────────
    {
        "text": "Pooja Nair: I am on my fourth consecutive week covering both the Cliq bot migration and the employee_scores rework alone. I am running on fumes.\nKaran Mehta: I noticed your commits are going in at midnight. Are you sleeping?\nPooja Nair: Three to four hours. The lead knows and said it will be looked at last week. Nothing changed.\nKaran Mehta: Then I am not raising mine either, what is the point.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Rohan Kapoor: The IIS data pipeline broke again and the only person who knows the fix is out on leave for a week.\nLakshmi Venkat: Who is picking it up?\nRohan Kapoor: Me. For the third time this month.\nLakshmi Venkat: Has anyone documented it at all?\nRohan Kapoor: No.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Aditya Sharma: Six releases this month and I am on on-call for three of them. I have not had a single uninterrupted weekend.\nNeha Gupta: Management promised a second on-call engineer last quarter.\nAditya Sharma: That person still has not been hired. I have stopped asking.\nNeha Gupta: Me too. Silence is safer.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Zara Hussain: I have been carrying the attrition dashboard rebuild since June. It is October and nobody has checked in.\nFaisal Khan: Did you flag the timeline?\nZara Hussain: Twice. It got marked as low priority and moved down.\nFaisal Khan: I am done speaking up too.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Devendra Singh: The Cliq message backfill is hitting hard limits and I have been debugging it since Monday. Five AM this morning again.\nSana Qureshi: That is unsustainable. What does the tech lead say?\nDevendra Singh: Said the deadline is non-negotiable. So here we are.\nSana Qureshi: I get it. I have been there.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Ishaan Reddy: I have rewritten the department sync endpoint seven times. Every time I think it is done, a new edge case surfaces.\nTara Krishnan: Are you getting any code review support?\nIshaan Reddy: My pull request has been sitting for eleven days.\nTara Krishnan: Same here with mine. We just ship and hope.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Nandini Patil: We are expected to deliver the client health scoring migration in half the time the original estimate gave.\nOm Joshi: What did planning say?\nNandini Patil: They said we committed to it. So I guess we are all working weekends now.\nOm Joshi: Again.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Kunal Verma: The token refresh implementation was dumped on my desk with zero handover and a Friday EOD deadline.\nRhea Iyer: Did you push back?\nKunal Verma: I tried. Got told it was already assigned and I just needed to execute.\nRhea Iyer: Then I will not be raising my workload either.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    {
        "text": "Priya Menon: I am handling the Zoho OAuth migration and the drift report simultaneously. It is not humanly possible to do both well.\nAmit Bansal: Management knows?\nPriya Menon: They approved the timeline without asking who was doing it. Two months ago.\nAmit Bansal: Then it is on them. We just break.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 0},
    },
    {
        "text": "Siddharth Rao: The only person who understood our sandbox fallback left last month and the knowledge transfer was never done.\nAnika Das: Are the docs updated?\nSiddharth Rao: There are no docs. I have been reverse engineering it for three weeks.\nAnika Das: I am sorry. I left before this got worse too.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 0},
    },
    # ── MODERATE (10) ─────────────────────────────────────────────────────
    {
        "text": "Varun Malhotra: The sprint got cut short again and I lost two days of planned work.\nDiya Srinivasan: Frustrating, but let us replan rather than absorb it silently.\nVarun Malhotra: You are right. Let me adjust the tickets and bring it up in planning.\nDiya Srinivasan: That is the way. The team has your back on scope.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Kavita Oberoi: I am not happy with how the channel analysis scope was decided, it felt rushed.\nNaveen Thakur: Fair. Do you want me to raise it with the PM together?\nKavita Oberoi: Yes please. I do not want to be the only one.\nNaveen Thakur: Done. We go in as a pair.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 1},
    },
    {
        "text": "Gautam Rathi: We are carrying a lot of context debt on the permissions module. It is starting to feel heavy.\nIsha Bhardwaj: I noticed that too. Let us schedule a cleanup sprint and get it refactored.\nGautam Rathi: I would welcome that. Just saying it out loud helps.\nIsha Bhardwaj: Always. That is what this channel is for.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Madhav Iyer: My review on the redirect config sat ignored for a week and that stung a bit.\nTanvi Kulkarni: Reopen it and tag me. I will make sure it gets looked at this time.\nMadhav Iyer: Appreciate that. I did not want to push back alone.\nTanvi Kulkarni: Never have to. That is the point.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 1},
    },
    {
        "text": "Saurabh Desai: Two full days lost to the deploy pipeline failing and I am frustrated we have not fixed the root cause yet.\nRitu Sharma: The team agrees. Let us block the next sprint until the pipeline is stable.\nSaurabh Desai: That would actually help. Sick of patching around it.\nRitu Sharma: I will propose it at planning.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Aarav Mehta: I keep getting pulled into client calls that are not mine. It is eating into my actual work.\nShreya Nair: Who owns those calls? Let me help you set a boundary with the account team.\nAarav Mehta: That would be useful. I have been afraid to say no.\nShreya Nair: You do not have to do it alone. Let us draft a message together.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 1},
    },
    {
        "text": "Nikhita Pillai: The RBAC rollout is behind and I feel responsible even though three of us are on it.\nYusuf Raza: The scope changed twice. It is not just you.\nNikhita Pillai: I know, but it is stressful watching it slip.\nYusuf Raza: Let us re-estimate and reset expectations. No one gets blamed for honest numbers.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Chetan Kulkarni: I am worried the drift thresholds we shipped are too aggressive and nobody wanted to hear it.\nLakshmi Rao: You were right and you know it. I will back you if it needs revisiting.\nChetan Kulkarni: Thanks. I did not want to be the one slowing the release.\nLakshmi Rao: Better a pause now than a rollback later.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "lack_of_support_and_isolation", "psychological_safety": 1},
    },
    {
        "text": "Snehal Shah: Two production incidents in a week and I am stretched thin between fixing and explaining.\nAarti Menon: You should not be doing both. Let me take the comms thread so you can focus.\nSnehal Shah: That would help a lot. The postmortems feel like a second job.\nAarti Menon: Consider it done. We share the load here.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": True, "root_cause": "overload_and_deadlines", "psychological_safety": 1},
    },
    {
        "text": "Raghav Iyengar: The daily standup keeps running long because we do not have a clear owner for the action items.\nPallavi Shetty: Good point. Want to propose a structured format and try it for a week?\nRaghav Iyengar: Yes. Happy to draft it.\nPallavi Shetty: Let us. Frustration without a plan is just noise.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 1},
    },
    # ── HEALTHY (10) ──────────────────────────────────────────────────────
    {
        "text": "Maya Brooks: The channel analysis panel shipped clean, nice work everyone.\nChris Park: The pagination fix from Imran made the difference.\nImran Shaikh: Happy to pair on the next one, I learned a lot from the review.\nMaya Brooks: That is exactly what I hoped for.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Deepa Iyer: I got the department sync wrong on my first pass and Nikhil caught it in review.\nNikhil Rao: Easy miss, the payload nests members differently than the docs suggest.\nDeepa Iyer: Documented it in the pull request so the next person does not repeat it.\nNikhil Rao: Perfect, that is how it should work.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Sam Ortiz: Sprint closed with everything done and no spillover for the first time this quarter.\nRiley Nguyen: The smaller scope really worked.\nSam Ortiz: Let us keep that shape next time.\nRiley Nguyen: Agreed, and I am genuinely looking forward to the simulation work.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Vikram Shetty: The aggregation now runs in under two seconds across the full scores collection.\nTejas Kulkarni: That is a tenfold improvement. How?\nVikram Shetty: Compound index plus dropping a redundant group stage. Writing it up now.\nTejas Kulkarni: Please do, I want the same trick on the client side.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Ananya Menon: I do not understand how the router picks a checkpoint, can someone walk me through it?\nFarhan Qureshi: Happy to, it is script detection before any inference runs. Fifteen minutes now?\nAnanya Menon: Yes please, that would help a lot.\nFarhan Qureshi: Grabbing a room.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Rahul Dubey: The permissions rollout went out with zero support tickets.\nMeera Joshi: Rohit's permission matrix doc did a lot of the heavy lifting there.\nRohit Varma: Thanks, and the testing pass from Divya caught two edge cases before release.\nDivya Nair: Team effort. Good one.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Kavya Reddy: I made a mistake on the redirect config and broke staging for an hour.\nArjun Bhat: Thanks for flagging it straight away, that is what matters.\nKavya Reddy: Added a check so it cannot happen silently again.\nArjun Bhat: Exactly the right response, no harm done.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Priya Shah: Retro actions from last time are all actually closed, which is new.\nJordan Lee: The shorter list helped. Three items instead of twelve.\nAlex Chen: I am happy to keep owning the deploy automation one.\nPriya Shah: Great, and shout if it needs a second pair of hands.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Sneha Pillai: Pairing on the drift detection engine all morning was the most useful thing I have done this month.\nImran Shaikh: Same, I finally understand the windowing logic.\nSneha Pillai: Let us make it a weekly slot.\nImran Shaikh: I am in.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
    {
        "text": "Tejas Kulkarni: The client portfolio demo went really well, the account managers were genuinely pleased.\nMeera Joshi: They asked for two small things, both easy.\nTejas Kulkarni: I will scope them tomorrow. Nice to end a week like this.\nMeera Joshi: Motivating, honestly.",
        "questions": QUESTIONS_BLOCK,
        "labels": {"burnout_risk": False, "root_cause": "none_healthy", "psychological_safety": 2},
    },
]


def main() -> None:
    with open(PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Appended {len(records)} records to {PATH}")


if __name__ == "__main__":
    main()
