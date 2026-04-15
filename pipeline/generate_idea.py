import argparse
import json
import os.path as osp
import os
import re
import traceback
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from json_repair import repair_json

import sys
sys.path.append(osp.join(osp.dirname(__file__), "../.."))

from utils.llm import AVAILABLE_LLMS, create_client, get_response_from_llm
from utils.memory_manager import MemoryManager
from tools.semantic_scholar import SemanticScholarSearchTool

from utils.prompt_all import BaseIdeaPrompt, IdeaDetails, AgentAction, TournamentResult
from utils.fix_latex import fix_latex_in_json


prompts = BaseIdeaPrompt()
system_prompt = prompts.system_prompt
idea_generation_prompt = prompts.idea_generation_prompt
idea_reflection_prompt = prompts.idea_reflection_prompt
evaluator_system_prompt = prompts.evaluator_system_prompt
evaluator_task_prompt = prompts.evaluator_task_prompt
# # Memory
memory = MemoryManager()
failed_memory = memory.get_failed_ideas()


# caching and tools
semantic_scholar_tool = SemanticScholarSearchTool()
CACHE_FILE = "../cache/semantic_scholar_cache.json"
cache_lock = asyncio.Lock()

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

search_cache = load_cache()

async def save_cache_async(cache_dict: dict):
    """Safely write to the cache file asynchronously to prevent corruption."""
    async with cache_lock:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=4)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def cached_semantic_scholar_search(query: str) -> str:
    """Wrapped Semantic Scholar search with in-memory caching and exponential backoff."""
    if query in search_cache:
        print(f"[Cache Hit] Semantic Scholar: '{query}'")
        return search_cache[query]

    # Assuming use_tool is synchronous; in a fully async app, make sure this doesn't block the event loop
    loop = asyncio.get_event_loop()
    raw_result = await loop.run_in_executor(None, semantic_scholar_tool.use_tool, query)

    # COMPRESSION: Prevent context bloat by extracting only top 5 relevant papers
    try:
        parsed_data = json.loads(raw_result)
        if isinstance(parsed_data, dict) and "data" in parsed_data:
            papers = parsed_data["data"][:10]  # Take only the top 10 papers
            compressed_results = []
            for p in papers:
                # Handle publisher elided abstracts elegantly
                abstract = p.get("abstract", "")
                if not abstract or "elided by the publisher" in abstract:
                    abstract = "[Abstract hidden by publisher. Rely on Title and Venue.]"
                compressed_results.append({
                    "Title": p.get("title", "Unknown"),
                    "Venue": p.get("venue", "Unknown"),
                    "Year": p.get("year", "Unknown"),
                    "Abstract": abstract
                })
            final_result_str = json.dumps(compressed_results, indent=2)
        else:
            final_result_str = str(raw_result)[:2000]  # Fallback truncation
    except Exception:
        final_result_str = str(raw_result)[:2000]  # Fallback if parsing fails

    search_cache[query] = final_result_str
    await save_cache_async(search_cache)

    return final_result_str


# # Async idea generator
async def generate_single_proposal(
        gen_idx: int,
        client: Any,
        model: str,
        workshop_description: str,
        prev_ideas_string: str,
        num_reflections: int
) -> Optional[Dict]:
    """Generates a single idea utilizing the reflection loop."""
    print(f"[Task {gen_idx}] Starting proposal generation...")

    last_tool_results = "No previous actions."
    msg_history = []

    for reflection_round in range(num_reflections):
        is_final_round = (reflection_round == num_reflections - 1)

        if reflection_round == 0:
            prompt_text = idea_generation_prompt.format(
                workshop_description=workshop_description,
                prev_ideas_string=prev_ideas_string,
                failed_memory=failed_memory
            )
        else:
            prompt_text = idea_reflection_prompt.format(
                current_round=reflection_round + 1,
                num_reflections=num_reflections,
                last_tool_results=last_tool_results,
            )

        if is_final_round:
            prompt_text += "\n\n*** URGENT: THIS IS YOUR FINAL ROUND. You are strictly forbidden from using 'SearchSemanticScholar' anymore. You MUST output the 'FinalizeIdea' action with your best possible proposal based on the context you have, otherwise the grant is lost. ***"

        # Add a hint about Semantic Scholar's elided abstracts
        prompt_text += "\n\nNote: If Semantic Scholar returns 'elided by the publisher: {abstract}', DO NOT keep searching for the same topic. Rely on the 'title' and 'venue' to infer context, or finalize your idea."

        try:
            loop = asyncio.get_event_loop()
            response_text, msg_history = await loop.run_in_executor(
                None, get_response_from_llm,
                prompt_text, client, model, system_prompt, msg_history
            )
            print(f"[Task {gen_idx} Round {reflection_round + 1}] Response text: {response_text}")
            # cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object could be extracted from the response.")

            raw_json_string = match.group(0)
            cleaned_response = repair_json(raw_json_string, return_objects=False)

            agent_output = AgentAction.model_validate_json(cleaned_response)

            print(f"[Task {gen_idx} Round {reflection_round + 1}] Action: {agent_output.action}")

            if agent_output.action == "SearchSemanticScholar":
                query = agent_output.action_arguments.get("query", "")
                last_tool_results = await cached_semantic_scholar_search(query)
                print(f"[Task {gen_idx}] Searched Semantic Scholar for: '{query}'")

            elif agent_output.action == "FinalizeIdea":
                # Extract the nested 'idea' dictionary from action_arguments
                idea_payload = agent_output.action_arguments.get("idea")
                if not idea_payload:
                    raise ValueError("Missing 'idea' wrapper inside action_arguments.")

                idea = IdeaDetails(**idea_payload)
                print(f"[Task {gen_idx}] Proposal finalized successfully: {idea.Name}")

                return {"idea": idea.model_dump(by_alias=True)}

        except ValidationError as e:
            last_tool_results = f"JSON parsing error: {e}. You MUST output strictly valid JSON. Fix this format bug."
            print(f"[Task {gen_idx}] Pydantic validation failed, ", last_tool_results)
        except Exception as e:
            last_tool_results = f"System Error: {str(e)}"
            print(f"[Task {gen_idx}] System Error: {str(e)}")
            traceback.print_exc()

    # Log the failure to memory
    if agent_output and agent_output.action_arguments.get("idea"):
        failed_idea = agent_output.action_arguments["idea"]
        memory.log_failed_idea(
            failed_idea.get("Name", "Unknown"),
            failed_idea.get("Short Hypothesis", ""),
            last_tool_results  # This contains the final Red Team critique
        )
    print(f"[Task {gen_idx}] Failed to finalize an idea within {num_reflections} rounds.")
    return None


async def evaluate_and_rank_ideas(ideas: list[dict], evaluated_idea: list[dict], client, model, max_retries: int=10):
    print("\n=======================================================")
    print("🏆 INITIATING IDEA TOURNAMENT (LLM-AS-A-JUDGE) 🏆")
    print(f"Evaluating {len(ideas)} candidate ideas...")

    # # Safely unwrap the {"idea": {...}} nesting
    unwrapped_ideas = [i.get("idea", i) for i in ideas]
    unwrapped_evaluated_ideas = [i.get("idea", i) for i in evaluated_idea]
    all_unwrapped_ideas = unwrapped_ideas + unwrapped_evaluated_ideas

    # if len(unwrapped_ideas) <= 1:
    #     print("   Only 1 idea generated. Defaulting as the winner.")
    #     unwrapped_ideas[0]["Tournament_Score"] = 30  # Default max score
    #     unwrapped_ideas[0]["Tournament_Justification"] = "Only one idea generated."
    #     return unwrapped_ideas[0]

    # Format the ideas for the prompt
    ideas_text = ""
    for idx, idea in enumerate(unwrapped_ideas):
        ideas_text += f"\n--- IDEA INDEX: {idx} ---\n"
        ideas_text += f"Title: {idea.get('Title')}\n"
        ideas_text += f"Hypothesis: {idea.get('Short Hypothesis')}\n"
        ideas_text += f"Limitations: {idea.get('Limitations')}\n"

    prompt_text = evaluator_task_prompt.format(num_ideas=len(unwrapped_ideas), ideas_text=ideas_text)
    msg_history = []

    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            response_text, _ = await loop.run_in_executor(
                None, get_response_from_llm, prompt_text,
                client, model, evaluator_system_prompt, msg_history
            )
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object could be extracted.")

            cleaned_response = match.group(0)
            cleaned_response = repair_json(cleaned_response, return_objects=False)
            cleaned_response = cleaned_response.replace('"index":', '"idea_index":')
            tournament_output = TournamentResult.model_validate_json(cleaned_response)

            print("\n--- TOURNAMENT RESULTS ---")
            # Inject the scores and justifications into our original dictionaries
            for eval_score in tournament_output.evaluations:
                idx = eval_score.idea_index
                if 0 <= idx < len(unwrapped_ideas):
                    unwrapped_ideas[idx]["Tournament_Score"] = eval_score.total_score
                    unwrapped_ideas[idx]["Tournament_Justification"] = eval_score.justification
                print(f"Idea {idx} | Score: {eval_score.total_score}/30 | {eval_score.justification}")

            # Sort the ideas list descending based on the Tournament_Score
            ranked_ideas = sorted(all_unwrapped_ideas, key=lambda x: x.get("Tournament_Score", 0), reverse=True)

            winning_idea = ranked_ideas[0]
            print(f"\n🎉 WINNER: {winning_idea['Title']}")

            return winning_idea, ranked_ideas

        except Exception as e:
            print(f"   ! Warning: Tournament parsing failed on attempt {attempt + 1}: {e}")
            # Feed the error back to the LLM so it corrects its JSON formatting
            prompt_text += f"\n\n*** CORRECTION REQUIRED: Your previous JSON failed validation: {e}. You MUST fix it. ***"

    print("Tournament evaluation failed max retries. Defaulting to Idea 0.")
    return unwrapped_ideas[0], all_unwrapped_ideas


async def generate_temp_idea(
        idea_fname: str,
        client: Any,
        model: str,
        workshop_description: str,
        max_num_generations: int = 20,
        num_reflections: int = 5,
        reload_ideas: bool = True,
) -> List[Dict]:
    idea_str_archive = []

    ini_idea_fname = idea_fname.replace(".json", "_initial_ideas.json")
    if osp.exists(ini_idea_fname):
        with open(ini_idea_fname, "r", encoding="utf-8") as f:
            idea_str_archive = json.load(f)
            print(f"Loaded {len(idea_str_archive)} generated_files from {ini_idea_fname}")
    else:
        print(f"No generated_files found in {ini_idea_fname}. Starting from scratch.")

    if len(idea_str_archive) > 5:
        prev_ideas_string = "\n\n".join([i['Abstract'] for i in idea_str_archive[:6]])
    else:
        prev_ideas_string = "\n\n".join([i['Abstract'] for i in idea_str_archive])
    print(f"Previous ideas string: {prev_ideas_string}")

    # Run generations concurrently using asyncio.gather
    tasks = [
        generate_single_proposal(
            gen_idx=i + 1,
            client=client,
            model=model,
            workshop_description=workshop_description,
            prev_ideas_string=prev_ideas_string,
            num_reflections=num_reflections
        )
        for i in range(max_num_generations)
    ]

    results = await asyncio.gather(*tasks)

    if len(idea_str_archive) > 20:
        idea_evaluated = idea_str_archive[:20]
    else:
        idea_evaluated = idea_str_archive

    # Filter out None values (failed generations)
    new_ideas = [res for res in results if res is not None]

    winning_idea, ranked_ideas = await evaluate_and_rank_ideas(new_ideas, idea_evaluated, client, model)
    # Save ONLY the winning idea to the JSON so the rest of the pipeline uses it
    final_output = [{"idea": winning_idea}]

    with open(idea_fname, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
    print(f"Stored the best idea in {idea_fname}")

    with open(ini_idea_fname, "w", encoding="utf-8") as fi:
        json.dump(ranked_ideas, fi, indent=4)

    print(f"Stored {len(new_ideas)} generated initial ideas and the previous {len(ranked_ideas)-len(new_ideas)} ideas in {ini_idea_fname}")
    return idea_str_archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AI collaborator proposals"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-reasoner",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=2,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--workshop-file",
        type=str,
        default="generated_files/new_idea_sgv.md",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=1,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    with open(args.workshop_file, "r", encoding="utf-8") as f:
        workshop_description = f.read()
    print(f"Using workshop description from {args.workshop_file} for idea generation.")
    print(f"Workshop description:\n{workshop_description}")

    # Create output filename by replacing .md extension with .json
    idea_fname = args.workshop_file.replace(".md", ".json")
    print("Starting idea generation for", idea_fname)
    ideas = asyncio.run(
        generate_temp_idea(
            idea_fname=idea_fname,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            max_num_generations=args.max_num_generations,
            num_reflections=args.num_reflections,
        )
    )
    print(f"{args.workshop_file} has generated {len(ideas)} in total.")
