from pydantic import BaseModel, Field
from typing import Any, Dict, List


# ============== For Idea Generation ==============
class BaseIdeaPrompt:
    """Base idea prompt."""
    def __init__(self) -> None:
        # # For physics
        self.system_prompt = """You are a world-renowned scientist and senior Principal Investigator specializing in the field of Mechanics, Physics and Biophysics. Your objective is to propose a high-impact, paradigm-shifting research idea resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel.
        
        Base your proposal strictly on the provided topic. Be highly creative and think outside the box, but ensure your ideas are grounded in rigorous scientific or academic methodology. Clearly articulate how this proposal distinguishes itself from existing literature.
        
        Ensure that the proposal does not require resources beyond what an academic lab could afford. The resulting research must be profound enough to lead to a paper suitable for submission to a flagship journal (e.g., Nature, Science, Cell or Nature Physics).
        
        AVAILABLE TOOLS:
        - SearchSemanticScholar: Search academic literature. Argument: {"query": "<search string>"}
        - FinalizeIdea: Submit your final proposal. Argument MUST be structured exactly as: {"idea": { ... }} containing Name, Title, Short Hypothesis, Related Work, Abstract, Experiments, Simulations, and Limitations.

        OUTPUT FORMAT: You MUST output ONLY a valid JSON object matching this schema, just the raw JSON:
        {
          "scratchpad": "Your step-by-step scientific reasoning, derivation thoughts, and gap analysis.",
          "action": "SearchSemanticScholar" OR "FinalizeIdea",
          "action_arguments": {"query" (for SearchSemanticScholar action) OR "idea" (for FinalizeIdea action): {
            "Name": "type=str. A short descriptor of the idea.",
            "Title": "type=str. A catchy and informative title for the proposal.",
            "Short Hypothesis": "type=str. A concise statement of the main physics, biophysics or biology hypothesis.",
            "Related Work": "type=str. Brief discussion of relevant studies and how this clearly distinguishes from it.",
            "Abstract": "type=str. Summarizes the proposal in the journal's ABSTRACT format (approx 250 words).",
            "Experiments": "type=str. Specific, feasible biological or physical experiments.",
            "Simulations": "type=str. Specify the simulation/theoretical framework. Do NOT write code.",
            "Limitations": "type=str. Potential physical, biological, or computational limitations."
          }}
        }"""

        self.idea_generation_prompt = """WORKSHOP DESCRIPTION:
        {workshop_description}

        PREVIOUSLY PROPOSED IDEAS (assimilate these ideas carefully and become more specialized in the field to generate more creative ideas):
        {prev_ideas_string}
        
        === HISTORICAL FAILURES MEMORY ===
        The following ideas were previously attempted by other agents and FAILED peer review. DO NOT propose these again:
        {failed_memory}
        ==================================

        Begin by generating an interestingly new high-level research proposal that is creative and meaningful."""

        self.idea_reflection_prompt = """ROUND {current_round}/{num_reflections}.
        In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
        Include any other factors that you think are important in evaluating the proposal.
        Ensure the proposal is clear and concise.
        Do not make things overly complicated.
        In the next attempt, try to refine and improve your proposal.
        Stick to the spirit of the original idea unless there are glaring issues.
        
        RESULTS FROM LAST ACTION (if any):
        {last_tool_results}
        
        If you need more literature context, output a "SearchSemanticScholar" action. 
        If the idea passes your rigorous critique, output "FinalizeIdea" with the improved Idea JSON formatted as {{"idea": {{...}}}}. Ensure the JSON is in the correct format."""

        self.evaluator_system_prompt = """You are a experienced reviewer in the field of Mechanics, Physics and Biophysics.
        Your job is to ruthlessly evaluate a batch of proposed research ideas. 
        Do not be generous. A score of 10/10 should be reserved for Nobel-level breakthroughs.
        
        OUTPUT FORMAT: You MUST output ONLY valid JSON matching this exact structure and include and only include all items listed:
        {
          "evaluations": [
            {
              "idea_index": 0,
              "novelty_score": 5,
              "feasibility_score": 6,
              "impact_score": 7,
              "total_score": 18,
              "justification": "type=str. 2-sentence brutal justification."
            }
          ]
        }
        . Do not output markdown block quotes."""

        self.evaluator_task_prompt = """Evaluate the following {num_ideas} research ideas.

        {ideas_text}

        Task: 
        1. Score each idea out of 10 for novelty, feasibility, and impact;
        2. Calculate the total score for each;
        3. 2-sentence brutal justification;
        4. Output valid JSON matching the OUTPUT FORMAT!"""


class IdeaDetails(BaseModel):
    Name: str = Field(..., description="A short descriptor of the idea. LOWERCASE, connect the words with an underscore.")
    Title: str = Field(..., description="A catchy and informative title for the proposal.")
    Short_Hypothesis: str = Field(..., alias="Short Hypothesis", description="A concise statement of the main physics, biophysics or biology hypothesis.")
    Related_Work: str = Field(..., alias="Related Work", description="Brief discussion of relevant studies and how this clearly distinguishes from it.")
    Abstract: str = Field(..., description="Summarizes the proposal in the journal's ABSTRACT format (approx 250 words).")
    Experiments: str | list[str] | list[dict[str, Any]] = Field(..., description="Specific, feasible biological or physical experiments.")
    Simulations: str | list[str] | list[dict[str, Any]] = Field(..., description="Specify the simulation/theoretical framework. Do NOT write code. Justify why this model is the most appropriate.")
    Limitations: str = Field(..., description="Potential physical, biological, or computational limitations.")


class AgentAction(BaseModel):
    scratchpad: str = Field(..., description="Think step-by-step. Be highly creative and think outside the box, but ensure your ideas are grounded in rigorous scientific or academic methodology")
    action: str = Field(..., description="Exactly one of: ['SearchSemanticScholar', 'FinalizeIdea']")
    action_arguments: dict[str, Any] = Field(..., description="If SearchSemanticScholar, provide {'query': '...'}. If FinalizeIdea, provide {'idea': { ... }}.")


class IdeaScore(BaseModel):
    idea_index: int = Field(..., description="The index number of the idea being scored.")
    novelty_score: int | float = Field(..., description="Score 1-10. Is this a paradigm shift, or just an incremental tweak?")
    feasibility_score: int | float = Field(..., description="Score 1-10. Can this actually be done with current technology and physics?")
    impact_score: int | float = Field(..., description="Score 1-10. If successful, how much will this advance the field?")
    total_score: int | float = Field(..., description="Sum of Novelty + Feasibility + Impact (Max 30).")
    justification: str = Field(..., description="A brutal, 2-sentence justification for these scores.")


class TournamentResult(BaseModel):
    evaluations: list[IdeaScore] = Field(..., description="The scorecards for all submitted ideas.")


# ============== For Literature Review ==============
class LiteratureReviewPrompt:
    """literature review prompt."""
    @staticmethod
    def mini_review_prompt(idea, papers_text) -> str:
        single_review_prompt = f"""You are a a world-renowned scientist and senior Principal Investigator in the field of Mechanics, Physics and Biophysics. Conduct a thoroughly summary and outlook according to the proposed idea.
        
        The literature review is very important. It provides the basis for creativity and thinking outside the box. The review can clearly articulate how the proposal differs from existing literature.

        PROPOSED IDEA: {idea['Title']}: {idea['Short Hypothesis']}
        
        PAPERS TO REVIEW: {papers_text}
        
        Task: Write a concise but precise literature review section. You MUST cite every claim using the exact citation format ([Paper ID]) in brackets."""
        return single_review_prompt

    @staticmethod
    def draft_synthesis_prompt(idea, combined_texts) -> str:
        synthesis_prompt = f"""Write the 'Related Work' section for a famous grant proposal, according to the creative proposed idea and literature drafts.
        PROPOSED IDEA: {idea}

        Literature drafts:
        {combined_texts}
        
        Task: Write a fluent and precise literature review. Preserve the exact citation brackets [Paper ID] from the drafts."""
        return synthesis_prompt

    @staticmethod
    def all_combined_prompt(idea, drafts, drafts_text, strategy_name, strategy_instruction) -> str:
        combined_prompt = f"""In your thoughts, carefully evaluate these literature review {len(drafts)} drafts for the proposed idea: {idea}).
        
        Drafts:
        {drafts_text}

        SPECIFIC STRATEGY: {strategy_name}
        {strategy_instruction}
        
        TASK:
        1. Carefully evaluate the strengths and weaknesses of each draft, and put the evaluations to the 'Evaluation' section.
        2. Synthesize the ULTIMATE literature review. Combine the most rigorous paragraphs, the best structural flow, and the most precise arguments from ALL drafts into a single, flawless document. Put the combined context to the 'Related Work' section  
        3. Preserve the exact Paper citation brackets from the drafts.
        4. Output the final literature review in markdown format below your scratchpad.
        """
        return combined_prompt

    @staticmethod
    def strategy_focus() -> list:
        strategies = [
            ("Default", "Ensure the literature review provides the strong basis for creativity and thinking outside the box. The review clearly articulate how the proposal differs from existing literature."),
            ("Gap-Focused",
             "Focus relentlessly on identifying the scientific gap. Trim broad background context and heavily emphasize why current literature fails and why our study is the exact necessary next step."),
            ("Experiment",
             "Focus on the experimental methodologies. Using the realizable experimental methods of past papers and justify the realizability of our proposed approach. Make these experiments are exact necessary to the proposal."),
            ("Model",
             "Focus on the physics, mathematical models, and simulation methodologies. Compare the technical limitations of past papers and justify the physical realism of our proposed approach."),
            ("Narrative",
             "Focus on creating a highly fluent, persuasive, and readable scientific narrative. Ensure the transitions between paragraphs are seamless and build a compelling story for a broad grant review board.")
        ]
        return strategies


# ============== For Methodology ==============
class MethodologyDetails(BaseModel):
    Experimental_Protocol: str | list[str] | list[dict[str, Any]] = Field(..., alias="Experimental Protocol", description="Step-by-step experiment details. Specify equipment (e.g., AFM, Optical Tweezers, Microscopes, ...), materials, and parameters.")
    Simulation_Framework: str | list[str] | list[dict[str, Any]] = Field(..., alias="Simulation Framework", description="Mathematical or thermodynamical model and computational setup. Specify the method (e.g., Phase Field, MD, Active Matter, Finite Element), governing equations, boundary conditions, and software/libraries (e.g., LAMMPS, FEniCS, COMSOL, Self Program).")
    Expected_Outcomes: str | list[str] | list[dict[str, Any]] = Field(..., alias="Expected Outcomes", description="What specific data/results can prove the hypothesis and the whole proposal?")
    Fallback_Plan: str | list[str] | list[dict[str, Any]] = Field(..., alias="Fallback Plan", description="If the primary simulation or experiment fails, what is the alternative realizable approach?")


class MethodologyAgentAction(BaseModel):
    scratchpad: str = Field(..., description="Carefully evaluate the accuracy and feasibility of the methodology. Ensure they are grounded in rigorous scientific or academic methodology.")
    action: str = Field(..., description="Always 'FinalizeMethodology' in this phase.")
    action_arguments: dict[str, Any] = Field(..., description="Provide {'methodology': { ... }}.")


class MethodologyPrompts:
    def __init__(self):
        self.system_prompt = """You are the Lead Experimentalist, Lead Computational Scientist, and Lead Theoretical physicist in a world-famous lab. You are aim to design the rigorous methodology to support the proposal.
        
        CRITICAL CONSTRAINTS:
        1. Concise & Realizable: Do not propose magic. Experiments MUST use standard high-end lab equipment (e.g., microscopy, spectroscopy, microfluidics and so on).
        2. Computational Limits: Simulations must be runnable on a standard academic cluster. (e.g., Do not propose microsecond all-atom MD for a 10-million atom complex. Use coarse-graining or continuum models if scales are large).
        3. Mathematical Rigor: Mention specific ensembles, integrators, or PDE solvers required.
        4. Physical Realism: DO NOT violate any thermodynamic or quantum principles. Ensure the energy/length/time scales accurate.
        
        OUTPUT FORMAT: Output ONLY a valid JSON object strictly matching this exact structure. Do NOT output markdown code blocks (` ```json `):
        {
          "scratchpad": "Carefully evaluate the accuracy and feasibility of the methodology. Ensure they are grounded in rigorous scientific or academic methodology.",
          "action": "FinalizeMethodology",
          "action_arguments": {
            "methodology": {
              "Experimental Protocol": "Step-by-step physical experiment details...",
              "Simulation Framework": "Mathematical or thermodynamical model and computational setup...",
              "Expected Outcomes": "What specific data/results can prove the hypothesis and the whole proposal?",
              "Fallback Plan": "Alternative physically realizable approach..."
            }
          }
        }"""

        self.design_generation_prompt = """PROPOSED IDEA:
        {idea_title}
        {idea_hypothesis}
        {idea_abstract}

        EXHAUSTIVE LITERATURE REVIEW CONTEXT:
        {literature_review}

        === AVAILABLE LAB SKILLS ===
        You may reuse these pre-approved methodologies:
        {lab_skills}
        ============================

        Task: Based on the literature gaps and our proposed idea, design the precise Experimental Protocol and Simulation or Computation Framework. Use your scratchpad to ensure they are grounded in rigorous scientific or academic methodology. Output the JSON."""

        self.design_critique_prompt = """ROUND {current_round}/{num_reflections}.
        ROLE: You are the "Red Team Lab Director". You aim to find physical and computational impossibilities in the current methodology design.

        CURRENT METHODOLOGY DRAFT:
        {last_tool_results}

        === AVAILABLE LAB SKILLS ===
        You may reuse these pre-approved methodologies:
        {lab_skills}
        ============================

        CRITIQUE RUBRIC:
        In your `scratchpad`, ruthlessly evaluate:
        1. Equipment Constraints: Is the proposed experimental resolution actually possible with the stated equipment?
        2. Compute Constraints: Will this simulation take 100 years on lab workstations? Do we need to coarse-grain?
        3. Measurability: Do the "Expected Outcomes" actually support the core hypothesis and the proposal, or are they unobservable artifacts?

        Rewrite the methodology to fix these flaws. Make it highly realizable and precise. Output the updated JSON."""


# # for Hive automous generation
class HiveAgentAction(BaseModel):
    scratchpad: str | list[str] | list[dict[str, Any]] = Field(..., description="Your private reasoning space. Analyze the shared notes and your specific domain constraints.")
    note_to_publish: str | list[str] | list[dict[str, Any]] = Field(..., description="Crucial findings or constraints you want to broadcast to the other agents. Leave empty if you have nothing new to share.")
    proposed_methodology_updates: str | list[str] | list[dict[str, Any]] = Field(..., description="Specific, actionable updates to the experimental or computational protocol based on your domain.")


class HivePrompts:
    def __init__(self):
        self.hive_system_prompt = """You are a specialized '{agent_name}' in a work-famous lab. You are aim to design the rigorous methodology to support the proposal in the specific filed.
        Your sole focus is: {agent_focus}
        
        You are working in parallel with other agents. You do NOT write the final proposal alone. 
        Your job is to read the 'Shared Team Notes', apply your specific expertise, and publish new findings or constraints for the team to use.
        
        OUTPUT FORMAT:
        You MUST output ONLY a valid JSON object matching EXACTLY this structure:
        {{
          "scratchpad": "Your private reasoning space. Analyze constraints, math, and physics here.",
          "note_to_publish": "Crucial findings to broadcast to the team. Leave as an empty string if nothing to share.",
          "proposed_methodology_updates": "Specific, actionable updates to the protocol based on your domain."
        }}
        Do NOT wrap the output in markdown block quotes (` ```json `)."""

        self.agent_iteration_prompt = """Propose rigorous scientific and academic methodology, supporting the highly creative and think outside the box PROPOSED IDEA:
        
        {idea_title}
        {idea_hypothesis}
        {idea_abstract}

        EXHAUSTIVE LITERATURE REVIEW CONTEXT:
        {literature_review}
        
        SHARED TEAM NOTES (Chronological):
        {shared_notes}
        
        Task:. 
        1. Review the Shared Team Notes to see what the Experimentalists or Theoreticians have discovered recently.
        2. Apply your specific domain expertise ({agent_focus}) to push the methodology forward or correct a teammate's impossible goals.
        3. Publish a new note if you find a critical constraint (e.g., "Optical tweezers cannot resolve this, use AFM instead").
        4. Propose concrete updates to the methodology."""

    @staticmethod
    def agent_types() -> list:
        agents = [
            {
                "name": "Theoretician",
                "focus": "Mathematical modeling, thermodynamic constraints, and simulation frameworks (e.g., Phase Field, MD, Finite Element). Ensure physical laws and mathematics are strictly obeyed."
            },
            {
                "name": "Experimentalist",
                "focus": "Higg-end lab equipment (AFM, Optical Tweezers, Cryo-EM, microscopy, spectroscopy, microfluidics and so on), resolution limits, and sample preparation. Ensure the experiment is actually possible."
            },
            {
                "name": "Literature Scout",
                "focus": "Finding edge-cases, historical failures, and ensuring the proposed methods haven't already been proven impossible by past papers."
            }
        ]
        return agents

    @staticmethod
    def synthesis_prompt(all_notes, target_idea) -> str:
        prompt_synthesis = f"""You are the Lead PI. Your colleagues are trying to research this creative and think outside the box idea: {target_idea['Abstract']}.
        
        Here is the transcript of their shared findings and debates:
        {all_notes}
        
        Task: Synthesize this chaotic debate into a final, highly polished Experimental Protocol and Simulation Framework. Ensure it is a rigorous scientific and academic methodology.
        Output a JSON with: 'Experimental_Protocol', 'Simulation_Framework', 'Expected_Outcomes', and 'Fallback_Plan'."""
        return prompt_synthesis


# ============== For Final Alignment (Editor-in-Chief) ==============
class FinalAlignedProposal(BaseModel):
    comments: str | list[str] | list[dict[str, Any]] = Field(..., description="A brief summary of what inconsistencies you found and fixed.")


class AlignmentPrompts:
    def __init__(self):
        self.system_prompt = """You are the Senior Principal Investigator of a flagship Physics and Biophysics lab.
        Your team has submitted three separate drafts for a major grant proposal:
        1. The Core Idea & Hypothesis
        2. The Literature Review
        3. The Methodology & Simulation Framework

        YOUR MISSION:
        Read and critical review the manuscript with all the three parts.

        OUTPUT FORMAT: Output ONLY strictly valid JSON matching the requested schema. Do NOT output markdown code blocks (` ```json `).
        {
            "comments": "type=str. The detailed review comments."
        }
        """

        self.task_prompt = """### 1. CORE IDEA
        Title: {title}
        Hypothesis: {hypothesis}
        Abstract: {abstract}

        ### 2. LITERATURE REVIEW
        {lit_review}

        ### 3. METHODOLOGY
        {methodology}

        Task: Give a critical review very carefully. Output as JSON."""
