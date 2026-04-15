import os
import json
import glob
from typing import List, Dict, Optional


class MemoryManager:
    """
    Manages the Shared Persistent Memory workspace for the AI Scientist.
    Based on the CORAL architecture: attempts/, notes/, and skills/.
    """

    def __init__(self, workspace_dir: str = "memory"):
        self.workspace_dir = workspace_dir
        self.attempts_dir = os.path.join(workspace_dir, "attempts")
        self.notes_dir = os.path.join(workspace_dir, "notes")
        self.skills_dir = os.path.join(workspace_dir, "skills")
        self._initialize_directories()

    def _initialize_directories(self):
        """Creates the directory structure if it doesn't exist."""
        for directory in [self.attempts_dir, self.notes_dir, self.skills_dir]:
            os.makedirs(directory, exist_ok=True)

    # ==========================================
    # ATTEMPTS: Tracking successes and failures
    # ==========================================
    def log_failed_idea(self, idea_name: str, hypothesis: str, failure_reason: str):
        """Logs an idea that Reviewer 2 rejected, so future agents avoid it."""
        filepath = os.path.join(self.attempts_dir, "failed_hypotheses.md")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"\n### {idea_name}\n**Hypothesis:** {hypothesis}\n**Why it Failed:** {failure_reason}\n---\n")

    def get_failed_ideas(self) -> str:
        """Reads historical failures to inject into the Generator prompt."""
        filepath = os.path.join(self.attempts_dir, "failed_hypotheses.md")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return "No past failures recorded yet."

    # ==========================================
    # NOTES: Caching literature & insights
    # ==========================================
    def save_paper_extract(self, paper_id: str, title: str, text: str):
        """Saves a Deep-Read PDF so we don't re-download it tomorrow."""
        filepath = os.path.join(self.notes_dir, f"{paper_id}.json")
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"paperId": paper_id, "title": title, "full_text": text}, f, indent=2)

    def load_cached_paper(self, paper_id: str) -> Optional[str]:
        filepath = os.path.join(self.notes_dir, f"{paper_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f).get("full_text", "")
        return None

    # ==========================================
    # SKILLS: Reusable Methodologies
    # ==========================================
    def save_skill(self, method_name: str, methodology_json: dict):
        """Saves a successfully vetted Simulation or Experimental Framework."""
        safe_name = method_name.replace(" ", "_").lower()
        filepath = os.path.join(self.skills_dir, f"skill_{safe_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(methodology_json, f, indent=4)

    def get_all_skills(self) -> str:
        """Retrieves all reusable skills to help the Methodology Designer."""
        skills = []
        for filepath in glob.glob(os.path.join(self.skills_dir, "*.json")):
            with open(filepath, "r", encoding="utf-8") as f:
                skill_data = json.load(f)
                skills.append(json.dumps(skill_data))
        return "\n\n".join(skills) if skills else "No skills developed yet."

    # post and read "Notes" on the shared bulletin board
    def publish_team_note(self, agent_name: str, iteration: int, content: str):
        """Allows an agent to broadcast a finding to the rest of the swarm."""
        safe_name = agent_name.replace(" ", "_").lower()
        filepath = os.path.join(self.notes_dir, f"note_{safe_name}_iter{iteration}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"**Author:** {agent_name}\n**Iteration:** {iteration}\n\n{content}")

    def read_all_team_notes(self) -> str:
        """Reads all published notes so an agent can catch up on team progress."""
        notes_content = []
        note_files = glob.glob(os.path.join(self.notes_dir, "note_*.md"))

        # Sort by modification time to read them chronologically
        note_files.sort(key=os.path.getmtime)

        for filepath in note_files:
            with open(filepath, "r", encoding="utf-8") as f:
                notes_content.append(f.read())

        if not notes_content:
            return "No team notes published yet."
        return "\n\n---\n\n".join(notes_content)
