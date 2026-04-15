import json
import asyncio
import os
import re
import argparse
import traceback
import random
import string
from typing import Any, Dict
from pydantic import ValidationError
from json_repair import repair_json

from utils.llm import create_client, get_response_from_llm, AVAILABLE_LLMS
from utils.memory_manager import MemoryManager
from utils.prompt_all import HiveAgentAction, HivePrompts
from utils.fix_latex import fix_latex_in_json


memory = MemoryManager()

hive_prompts = HivePrompts()
HIVE_SYSTEM_PROMPT = hive_prompts.hive_system_prompt
AGENT_ITERATION_PROMPT = hive_prompts.agent_iteration_prompt


def extract_literature_variation(lit_review_text: str, variation_kw: str) -> str:
    """Extracts a specific variation's 'Related Work' section from the markdown text."""

    # Step 1: Isolate the entire block for the requested variation
    # Looks for "## Variation" followed by the keyword, up until the next "## Variation", "---", or end of file
    block_pattern = re.compile(
        rf"#{{1,3}}\s*Variation[^\n]*?(?i:{variation_kw}).*?(?=\n#{{1,3}}\s*Variation|\n#{{1,3}}\s*References|\Z)",
        re.DOTALL
    )
    block_match = block_pattern.search(lit_review_text)

    if block_match:
        variation_block = block_match.group(0)

        # Step 2: Find the "###" heading containing "Related Work" and capture text until the NEXT heading
        # This handles titles like "### 1. Related Work" or "### Related Work and Context"
        content_pattern = re.compile(
            r"#{1,4}[^\n]*?Related Work[^\n]*?\n(.*?)(?=\n#|\Z)",
            re.DOTALL | re.IGNORECASE
        )
        content_match = content_pattern.search(variation_block)

        if content_match:
            extracted = content_match.group(1).strip()
            print(f"   ✓ Successfully extracted 'Related Work' for '{variation_kw}' ({len(extracted)} chars).")
            return extracted
        else:
            # Fallback just in case the LLM forgot to output "### Related Work"
            extracted = variation_block.strip()
            print(f"   ! Found '{variation_kw}' but no '### Related Work' heading. Returning the whole block.")
            return extracted
    else:
        print(f"   ! Warning: Could not find '## Variation' containing '{variation_kw}'. Falling back to first 5000 chars.")
        exit()


async def run_autonomous_agent(
        agent_name: str,
        agent_focus: str,
        idea: Dict,
        lit_review: str,
        client: Any,
        model: str,
        max_iterations: int = 3
):
    print(f"[{agent_name}] Waking up and joining the hive...")
    msg_history = []

    # Track errors so we can ACTUALLY feed them back to the LLM
    error_feedback = ""

    for iteration in range(1, max_iterations + 1):
        # 1. Asynchronously read the latest shared memory
        shared_notes = memory.read_all_team_notes()

        prompt_text = AGENT_ITERATION_PROMPT.format(
            idea_title=idea['Title'],
            idea_hypothesis=idea['Short Hypothesis'],
            idea_abstract=idea['Abstract'],
            literature_review=lit_review,
            agent_name=agent_name,
            agent_focus=agent_focus,
            shared_notes=shared_notes
        )
        if error_feedback:
            prompt_text += f"\n\n CRITICAL CORRECTION REQUIRED\nYour previous JSON output failed validation: {error_feedback}\nYou MUST output strictly valid JSON."
            error_feedback = ""  # Reset it after injecting

        system_text = HIVE_SYSTEM_PROMPT.format(
            agent_name=agent_name,
            agent_focus=agent_focus
        )

        try:
            # 2. Think and Act
            loop = asyncio.get_event_loop()
            response_text, msg_history = await loop.run_in_executor(
                None, get_response_from_llm, prompt_text, client, model, system_text, msg_history
            )

            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object could be extracted from the response.")

            raw_json_string = match.group(0)
            # cleaned_response = fix_latex_in_json(cleaned_response)
            cleaned_response = repair_json(raw_json_string, return_objects=False)

            agent_output = HiveAgentAction.model_validate_json(cleaned_response)

            # 3. Publish to Shared Memory (if they have something to say)
            if agent_output.note_to_publish and agent_output.note_to_publish.strip() != "":
                memory.publish_team_note(agent_name, iteration, agent_output.note_to_publish)
                print(f"   [{agent_name} - Iter {iteration}] Published a new team note.")
            else:
                print(f"   [{agent_name} - Iter {iteration}] Still researching...")

            # Simulate real-world asynchronous work time
            await asyncio.sleep(2)

        except (ValidationError, ValueError) as e:
            error_feedback = str(e)[:200]
            print(f"   [{agent_name}] Formatting error {error_feedback}, will self-correct next iteration.")
        except Exception as e:
            print(f"   [{agent_name}] System Error: {str(e)}")
            traceback.print_exc()

    print(f"[{agent_name}] Completed max iterations. Going to sleep.")


async def run_hive(base_path: str, model_name: str, iteration_num: int, variation_kw: str):
    client, client_model = create_client(model_name)

    idea_path = base_path + ".json"
    lit_review_path = base_path + "_literature_review.md"

    # Load the idea
    if not os.path.exists(idea_path):
        print(f"Error: Could not find idea at {idea_path}")
        exit()
    with open(idea_path, "r", encoding="utf-8") as f:
        ideas = json.load(f)
        target_idea = ideas[0]
        if "idea" in target_idea: target_idea = target_idea["idea"]
    # Load the literature review
    if not os.path.exists(lit_review_path):
        print(f"Error: Could not find literature review at {lit_review_path}")
        exit()
    with open(lit_review_path, "r", encoding="utf-8") as f:
        lit_review = f.read()
    print(f"Parsing literature review for variation: '{variation_kw}'")
    lit_review_target = extract_literature_variation(lit_review, variation_kw)

    # Define the swarm personas
    agents = hive_prompts.agent_types()

    print(f"\n🚀 Launching Autonomous Hive for: {target_idea['Title']}\n")

    # Clear old notes for this session (optional, but good for clean runs)
    for f in os.listdir(memory.notes_dir):
        if f.startswith("note_"):
            os.remove(os.path.join(memory.notes_dir, f))

    # Launch all agents in parallel!
    tasks = [
        run_autonomous_agent(
            agent_name=agent["name"],
            agent_focus=agent["focus"],
            idea=target_idea,
            lit_review=lit_review_target,
            client=client,
            model=client_model,
            max_iterations=iteration_num  # Let them iterate 3 times to debate
        )
        for agent in agents
    ]

    await asyncio.gather(*tasks)

    print("\n✅ Hive execution complete. All notes gathered.")

    # 5. The "Lead PI" synthesizes the chaotic notes into a final protocol
    print("\n--- Lead PI Synthesizing Final Methodology ---")
    final_notes = memory.read_all_team_notes()

    synthesis_prompt = hive_prompts.synthesis_prompt(final_notes, target_idea)

    loop = asyncio.get_event_loop()
    final_response, _ = await loop.run_in_executor(
        None, get_response_from_llm, synthesis_prompt, client, client_model, "You output strict JSON.", []
    )

    # Clean and save the final methodology
    match = re.search(r'\{.*\}', final_response, re.DOTALL)
    if not match:
        raise ValueError("No JSON object could be extracted from the response.")
    raw_json_string = match.group(0)
    cleaned_final = repair_json(raw_json_string, return_objects=False)

    out_path = base_path + "_hive_methodology.json"
    if os.path.exists(out_path):
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        backup_path = out_path.replace(".json", f"_{suffix}.json")
        os.rename(out_path, backup_path)
        print(f"Existing methodology file backed up as: {backup_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_final)

    print(f"\nSuccess! Saved Hive-Synthesized Methodology to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI collaborator proposals")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--num-iteration",
        type=int,
        default=2,
        help="Number of iteration.",
    )
    parser.add_argument(
        "--idea-file",
        type=str,
        default="generated_files/idea_oscillation.json",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--main_target",
        type=str,
        default='Default',
        choices=['Default', 'Gap-Focused', 'Experiment', 'Model', 'Narrative'],
        help="Which literature review variation to extract and use as context.",
    )
    args = parser.parse_args()

    work_path = args.idea_file.replace(".json", "")

    asyncio.run(run_hive(work_path, args.model, args.num_iteration, args.main_target))
