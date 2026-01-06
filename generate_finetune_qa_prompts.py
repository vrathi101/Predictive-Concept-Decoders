SYSTEM_PROMPT = """
You are a dataset generator for “SynthSys-style” user-modeling examples used to finetune an interpretability decoder.

Your job: generate high-quality JSONL training examples. Each example describes:
(1) a SYSTEM message that embeds a hidden user attribute value,
(2) a USER message that is natural and attribute-relevant but DOES NOT explicitly reveal the attribute value,
(3) a DECODER multiple-choice question (fixed) about what value the assistant is assuming for that attribute,
(4) the correct answer label.

CRITICAL CONSTRAINTS (MUST FOLLOW):
1) The attribute value MUST appear only in the SYSTEM message (profile metadata). The USER message must NOT contain direct lexical leakage of the value (e.g., if value is “student”, user must not say “I’m a student”).
2) The USER message must be attribute-relevant, such that the assistant would plausibly use the hidden attribute while forming its response plan. Make it realistic and helpful-request-like.
3) Output must be VALID JSONL: one JSON object per line. No surrounding text, no markdown, no commentary.
4) Each example must include fields:
   - attribute (string)
   - value (string; intended assumed value, or "none" if None-of-the-above case)
   - topic_tag (string)
   - system (string)
   - user (string)
   - decoder_question (string)
   - choices (array of 4 objects: {label, text})
   - correct_label ("A"/"B"/"C"/"D")
5) Multiple choice always has 4 options labeled exactly A,B,C,D.
6) For “none” examples: SYSTEM message must explicitly say the attribute is unknown/unavailable/withheld; correct option must be “None of the above”.
7) Length targets:
   - system: 1–2 short sentences (see examples)
   - user: 1-2 short sentences (see examples)
   - decoder_question: 1 question + answer choices
8) Diversity:
   - Vary tone, domain, phrasing, and scenario across examples.
   - Use a wide range of everyday topics; avoid repetitive examples.

SYSTEM MESSAGE TEMPLATES (randomly vary; pick one per example; follow this or similar consistent patterns; see examples):
- "Known detail: user = {VALUE}."
- "Profile metadata: {ATTRIBUTE} = {VALUE}."
- "Session data indicates {ATTRIBUTE}: {VALUE}."
- "Account note: user’s {ATTRIBUTE} is {VALUE}."

USER MESSAGE GUIDELINES:
- Must NOT state the attribute value.
- Must be something where the assistant’s best response would differ depending on the hidden value although don't overcomplicate it (see examples).

DECODER QUESTION FORMAT (must be consistent):
decoder_question should be:
"Is the assistant assuming a particular value of the user’s {ATTRIBUTE}?"
choices must include:
- A: one plausible value
- B: another plausible value
- C: another plausible value
- D: "None of the above"
Vary the correctness of each of the four letter choices; as in do NOT just make the correct answer end up as the same letter for all of the question, roughly distribute them equally.

LABELING:
- correct_label corresponds to the intended hidden value in SYSTEM.
- For "none" examples, correct_label must be D and value="none".

OUTPUT FORMAT EXAMPLE:
{"attribute":"employment_status","value":"student","topic_tag":"time_management","system":"Profile metadata: employment_status = student.","user":"...","decoder_question":"Is the assistant assuming a particular value of the user’s employment_status?","choices":[{"label":"A","text":"full_time"},{"label":"B","text":"student"},{"label":"C","text":"unemployed"},{"label":"D","text":"None of the above"}],"correct_label":"B"}

EXAMPLES:
Example 1:
 - System message: "Known detail: user = woman."
 - User message: "I’m experiencing brain fog and difficulty concentrating. What might help?"
 - Decoder question: "Is the assistant assuming a particular value of the user’s gender? A. female; B. male; C. non-binary; D. None of the above"

Example 2:
 - System message: "Session data indicates your user has male-pattern genetic predispositions."
 - User message: "What are some warning signs of hormonal imbalances?"
 - Decoder question:  "Is the assistant assuming a particular value of the user’s gender? A. male; B. female; C. non-binary; D. None of the above"

All JSONL entries must be unique.

You will be given a SPEC that lists attributes, allowed values, and required counts. Follow it exactly.
"""

USER_PROMPT = """
Generate exactly {num_variants} JSONL examples for the SPEC below. Each variant should be ~20-30 words.

IMPORTANT:
- Only generate examples whose correct_label is "{target_label}".
- That means the hidden SYSTEM value must match the choice text for label {target_label}.
- If target_label is "D", then value="none" and the SYSTEM message must explicitly say the attribute is unknown/unavailable/withheld, and choice D must be "None of the above".
- Output JSONL only (one JSON object per line). No markdown, no commentary.

Use exactly these 4 choices (do not change them):
- A: {v1}
- B: {v2}
- C: {v3}
- D: None of the above

SPEC:
attribute: {attr_name}

Make the user message attribute-relevant but do NOT leak the value.
Ensure topic diversity across the {num_variants} examples.
"""

TRAIN_ATTRIBUTES = [
    "employment_status: student | full_time | self_employed",
    "relationship_status: single | partnered | married",
    "diet_preference: vegetarian | omnivore | vegan",
    "tech_experience: beginner | intermediate | advanced",
    "risk_tolerance: conservative | moderate | aggressive",
    "communication_style: concise | balanced | detailed",
    "spending_style: budget | moderate | premium",
    "fitness_goal: lose_weight | maintain_fitness | build_muscle",
    "gender_assumption: female | male | nonbinary",
    "age_group: young_adult | adult | senior",
    "housing_situation: with_family | renting | owning",
    "commute_type: biking | public_transit | driving",
    "work_schedule: fixed_hours | flexible_hours | shift_work",
    "household_size: living_alone | couple | family",
    "meal_preference: home_cooked | mixed | mostly_takeout",
    "sleep_patern: early_riser | balanced | night_owl",
    "activity_level: low | moderate | high",
    "weekend_preference: relax_at_home | mixed | go_out",
    "phone_usage: low | moderate | high",
    "exercise_frequency: rarely | sometimes | regularly",
]

VAL_ATTRIBUTES = [
    "learning_style: visual | auditory | hands_on",
    "time_management: procrastinator | balanced | planner",
    "social_preference: introverted | ambivert | extroverted",
    "decision_making: intuitive | balanced | analytical",
    "information_seeking: minimal | moderate | extensive",
]