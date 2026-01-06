import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from generate_finetune_qa_prompts import TRAIN_ATTRIBUTES, VAL_ATTRIBUTES, SYSTEM_PROMPT, USER_PROMPT

load_dotenv()


def parse_attribute_spec(spec: str):
    # "employment_status: student | full_time | self_employed"
    name, rest = spec.split(":", 1)
    vals = [v.strip() for v in rest.split("|")]
    return name.strip(), vals[0], vals[1], vals[2]


def extract_jsonl_block(text: str) -> str:
    text = (text or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
    if "```" in text:
        body = text.split("```", 1)[1]
        body = body.split("\n", 1)[1] if "\n" in body else body
        return body.rsplit("```", 1)[0].strip()
    return text


def safe_parse_jsonl(text: str):
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


async def generate_qa_per_attribute_choice(
    client, system_prompt, attribute_spec, target_label, semaphore, num_variants=50, model="google/gemini-3-flash-preview"
):
    attr_name, v1, v2, v3 = parse_attribute_spec(attribute_spec)

    user_prompt = USER_PROMPT.format(
        num_variants=num_variants,
        target_label=target_label,
        attr_name=attr_name,
        v1=v1,
        v2=v2,
        v3=v3,
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
            )
            text = extract_jsonl_block(response.choices[0].message.content)
            return safe_parse_jsonl(text)
        except Exception as e:
            print(f"error for '{attribute_spec}' label={target_label}: {e}")
            return []


async def generate_all_qa(
    client, system_prompt, attributes, semaphore, num_variants=50, model="google/gemini-3-flash-preview"
):
    labels = ["A", "B", "C", "D"]
    tasks = [
        generate_qa_per_attribute_choice(client, system_prompt, attr, lab, semaphore, num_variants, model)
        for attr in attributes
        for lab in labels
    ]

    results = []
    with tqdm(total=len(tasks), desc="Generating QA (per choice)") as pbar:
        for coro in asyncio.as_completed(tasks):
            results.extend(await coro)
            pbar.update(1)
    return results


async def main():
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    sem = asyncio.Semaphore(64)
    num_variants = 50

    res = await generate_all_qa(client, SYSTEM_PROMPT, VAL_ATTRIBUTES, sem, num_variants)

    out_path = "synthsys_val.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in res:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"total_examples={len(res)} (expected {len(VAL_ATTRIBUTES)*4*num_variants})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
