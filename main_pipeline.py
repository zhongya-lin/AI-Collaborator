import argparse
import asyncio
import json
import os
import re

from utils.llm import create_client, AVAILABLE_LLMS

from pipeline.generate_idea import generate_temp_idea
from pipeline.literature_review_engine import run_exhaustive_review
from pipeline.autonomous_hive import run_hive


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

# ==========================================
# PHASE 4: FINAL GRANT COMPILATION
# ==========================================
async def compile_final_proposal(base_path: str, client, model: str, lit_variation: str = "Gap-Focused"):
    print("\n=======================================================")
    print("PHASE 4: COMPILING MASTER GRANT PROPOSAL")
    print("=======================================================\n")

    idea_path = base_path + ".json"
    lit_path = base_path + "_literature_review.md"
    method_path = base_path + "_hive_methodology.json"
    final_out_path = base_path + "_final_proposal.md"

    # 1. Load all separated artifacts
    try:
        with open(idea_path, "r", encoding='utf-8') as f:
            target_idea = json.load(f)[0].get("idea", {})

        with open(lit_path, "r", encoding='utf-8') as f:
            full_lit_review = f.read()

        with open(method_path, "r", encoding='utf-8') as f:
            methodology = json.load(f)
    except FileNotFoundError as e:
        print(f"Compilation failed. Missing artifact: {e}")
        return

    # Extract the requested literature variation from the massive markdown file
    chosen_lit_review = extract_literature_variation(full_lit_review, lit_variation)

    # 2. Write the combined Markdown document
    print("Drafting final cohesive document...")

    with open(final_out_path, "w", encoding='utf-8') as f:
        # Title & Abstract
        f.write(f"# {target_idea.get('Title', 'Untitled Proposal')}\n\n")
        f.write(f"**Short Name:** {target_idea.get('Name', 'N/A')}\n\n")
        f.write("## Executive Abstract\n")
        f.write(f"{target_idea.get('Abstract', 'No abstract provided.')}\n\n")

        # Hypothesis
        f.write("## Core Hypothesis\n")
        f.write(f"> {target_idea.get('Short Hypothesis', '')}\n\n")

        # Literature Review (The chosen variation)
        f.write("## Literature Review & Scientific Gap\n")
        f.write(f"{chosen_lit_review}\n\n")

        # Methodology
        f.write("## Methodology & Research Plan\n")
        f.write("### 1. Experimental Protocol\n")
        f.write(f"{methodology.get('Experimental_Protocol', '')}\n\n")
        f.write("### 2. Simulation & Computational Framework\n")
        f.write(f"{methodology.get('Simulation_Framework', '')}\n\n")
        f.write("### 3. Expected Outcomes\n")
        f.write(f"{methodology.get('Expected_Outcomes', '')}\n\n")
        f.write("### 4. Risk Mitigation & Fallback Plan\n")
        f.write(f"{methodology.get('Fallback_Plan', '')}\n\n")

        # Limitations (From Idea phase)
        f.write("## Overall Limitations\n")
        f.write(f"{target_idea.get('Limitations', '')}\n\n")

    print(f"✅ Success! Master Grant Proposal saved to: {final_out_path}")


# ==========================================
# MASTER PIPELINE ORCHESTRATION
# ==========================================

async def run_full_pipeline(args):
    client, client_model = create_client(args.model)
    # Setup base path (e.g., ideas/idea_background)
    base_path = args.idea_file.replace(".md", "")

    with open(args.idea_file, "r", encoding="utf-8") as f:
        workshop_description = f.read()

    print("\n" + "=" * 55)
    print("🚀 INITIALIZING AI SCIENTIST PIPELINE")
    print("=" * 55)

    # ---------------------------------------------------------
    # PHASE 1: IDEA GENERATION
    # ---------------------------------------------------------
    idea_fname = base_path + ".json"
    if not os.path.exists(idea_fname) or args.force_restart:
        print("\n--- PHASE 1: Generating Novel Ideas and Hypotheses ---")
        await generate_temp_idea(
            idea_fname=idea_fname,
            client=client,
            model=client_model,
            workshop_description=workshop_description,
            max_num_generations=args.max_num_generations,
            num_reflections=args.num_reflections,
            reload_ideas=not args.force_restart
        )
    else:
        print("\n--- PHASE 1: Skipped (Idea JSON already exists) ---")

    # ---------------------------------------------------------
    # PHASE 2: EXHAUSTIVE LITERATURE REVIEW
    # ---------------------------------------------------------
    lit_fname = base_path + "_literature_review.md"
    if not os.path.exists(lit_fname) or args.force_restart:
        print("\n--- PHASE 2: Graph Traversal & Literature Review ---")
        await run_exhaustive_review(
            idea_path=idea_fname,
            model_name=args.model,
            num_drafts=args.max_num_reviews,
            citation_num=args.citation_num,
            main_target=args.main_target)
    else:
        print("\n--- PHASE 2: Skipped (Literature Review already exists) ---")

    # ---------------------------------------------------------
    # PHASE 3: AUTONOMOUS METHODOLOGY HIVE
    # ---------------------------------------------------------
    method_fname = base_path + "_hive_methodology.json"
    if not os.path.exists(method_fname) or args.force_restart:
        print("\n--- PHASE 3: Launching Autonomous Swarm ---")
        await run_hive(
            base_path=base_path,
            model_name=args.model,
            iteration_num=args.num_iteration,
            variation_kw=args.main_target
        )
    else:
        print("\n--- PHASE 3: Skipped (Methodology already exists) ---")

    # ---------------------------------------------------------
    # PHASE 4: COMPILATION
    # ---------------------------------------------------------
    await compile_final_proposal(
        base_path=base_path,
        client=client,
        model=client_model,
        lit_variation=args.main_target
    )

    # # ---------------------------------------------------------
    # # PHASE 5: LATEX TYPESETTING & PDF COMPILATION
    # # ---------------------------------------------------------
    # print("\n--- PHASE 5: Autonomous LaTeX Typesetting ---")
    # await generate_and_compile_latex(base_path=base_path, client=client, model=client_model)

    print("\n🎉 PIPELINE COMPLETE. THE AI SCIENTIST HAS FINISHED ITS WORK.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end AI Scientist Pipeline.")
    parser.add_argument("--idea-file", type=str, default="generated_files/idea_oscillation.md",
                        help="Path to the initial idea file.")
    parser.add_argument("--model", type=str, default="deepseek-reasoner",
                        choices = AVAILABLE_LLMS, help="Model to use.")
    parser.add_argument("--max-num-generations", type=int, default=5,
                        help="How many distinct ideas to generate.")
    parser.add_argument("--max-num-reviews", type=int, default=3,
                        help="How many distinct literature reviews to generate.")
    parser.add_argument("--num-reflections", type=int, default=10,
                        help="Number of self-correction rounds.")
    parser.add_argument("--citation-num", type=int, default=16,
                        help="Maximum number of citations per proposal in literature review generation.")
    parser.add_argument("--main_target", type=str, default="Gap-Focused",
                        choices=['Default', 'Gap-Focused', 'Experiment', 'Model', 'Narrative'],
                        help="Which literature to generate and use in the final document.")
    parser.add_argument("--num-iteration", type=int, default=5,
                        help="Number of iteration for methodology generation.")

    # Adding a flag to force a clean run
    parser.add_argument("--force-restart", type=bool, default=True,
                        help="If True, ignores cached files and runs the whole pipeline from scratch.")

    args = parser.parse_args()

    # Verify API key exists before starting a massive run
    if not os.getenv("S2_API_KEY"):
        print("⚠️  WARNING: 'S2_API_KEY' environment variable not found. Literature review will run slowly to avoid rate limits.")

    asyncio.run(run_full_pipeline(args))
