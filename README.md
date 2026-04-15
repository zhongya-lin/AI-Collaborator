# AI-Collaborator

> An Autonomous, Human-in-the-Loop Multi-Agent Swarm for Scientific Discovery in the Physical Sciences.

The **AI-Collaborator** is an open-source framework designed to augment the research pipeline for the physical and biological sciences (e.g., Biophysics, Materials Science, Wet-Lab Biology). 

Unlike existing AI systems that are confined to purely digital disciplines (software engineering, machine learning sandboxes), the AI-Collaborator acts as an elite **Principal Investigator Co-Pilot**. It autonomously executes the cognitive heavy-lifting of the scientific method, from hypothesis generation to methodology design, and outputs a fully typeset LaTeX manuscript ready for human review and physical laboratory execution.

## ✨ Key Features

- **🏆 Evolutionary Hypothesis Tournament (LLM-as-a-Judge):** Generates a batch of novel ideas and ruthlessly ranks them based on Novelty, Feasibility, and Impact before proceeding.
- **🕸️ Graph-Traversal Literature Review:** Integrates with the Semantic Scholar API (with robust exponential backoff) to exhaustively map related work and identify critical gaps.
- **🐝 The Asynchronous Hive:** A multi-agent swarm featuring a *Theoretician*, *Experimentalist*, and *Literature Scout*. They share persistent memory to debate and refine physically realizable methodologies.
- **📝 Peer Reviews:** Automatically review the entire manuscript.

---

## 🏗️ System Architecture

The pipeline executes in five sequential phases:

1. **Phase 1: Ideation & Evaluation.** Generates multiple hypotheses based on your seed topic and ranks them using scores given by AI.
2. **Phase 2: Literature Synthesis.** Searches the academic graph and writes a comprehensive, gap-focused literature review.
3. **Phase 3: Swarm Methodology.** The Hive agents debate mathematically sound simulation frameworks and realistic experimental protocols.
4. **Phase 4: Comments.** The 'AI Peer reviewers' give some comments.

---

## 🛠️ How to Use

1. Install Python Dependencies:

Python version >=3.10

```Bash
pip install -r requirements.txt
```

2. Set API Keys

Add a `.env` file to the root directory of the project and provide the LLM API key. This is required for the LLM agents and the Semantic Scholar key.

```
S2_API_KEY=xxxxxxx
DEEPSEEK_API_KEY=xxxxxxxx
```

3. Modify prompts:

All prompts are in utils/prompt_all.py. You can customise them according to your preferred field. The **AI-Collaborator** can be easily extended to other research fields, such as material science and engineering. MAKE SURE the format does not change. Just modify the words and phrases. 

Semantic Scholar API Key (Optional but recommended): Prevents severe rate-limiting during the literature review phase. You can request a free key from Semantic Scholar.

4. The Seed Topic

Add a markdown file. For the best results, use our highly structured template (`generated_files/idea_sgv.md`) and ensure that the Markdown file is placed in the folder `generated_files`.

The markdown file includes:

```Markdown
# Title: [Your Topic Here]

## Keywords
[keyword1, keyword2, keyword3]

## TL;DR
[One or two sentence summary of the research direction.]

## Abstract
[Detailed background, literature context, and core problem to solve...]
```

5. Resume / Force Restart

If the API crashes or you want to swap the winning idea, simply set "--force_restart" as `False` in `main_pipeline.py`. The pipeline will instantly pick up exactly where it left off by reading the cached JSON files in the /ideas directory.

6. Run:

```Bash
python3 main_pipeline.py
```

## 🧠 LLM Engines

Different LLMs excel at different parts of this pipeline. I just tested `deepseek-reasoner` and the local LLM through ollama `ollama/gemma4` (12/04/2026).

## 📄 Output Artifacts

Upon completion, the AI-Collaborator will generate the following files in the `generated_files/` directory:

`idea.json`: The top one ranked idea/hypothesis.

`initial_ideas.json`: All ranked ideas/hypotheses.

`literature_review.md`: The exhaustively cited background context.

`hive_methodology.json`: The raw agent debates and final protocols.

`final_proposal.md`: The combined markdown manuscript with 'AI reviewer' comments.

## Possible Bugs

-[ ] According to the keywords, there is no related paper in semantic scholar. Now, maybe you should open the `idea.json` to modify the `Name` and `Title` to make the keywords are meanful for search engine.


