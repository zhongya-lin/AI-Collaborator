import json
import string
import asyncio
import aiohttp
import torch
import re
import os
import argparse
import random
import fitz  # PyMuPDF
from typing import List, Dict, Set
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
nltk.download('punkt', quiet=True)       # Needed for splitting words
nltk.download('brown', quiet=True)       # Needed for noun phrase extraction
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

from utils.llm import create_client, get_response_from_llm, AVAILABLE_LLMS
from utils.memory_manager import MemoryManager

from utils.prompt_all import LiteratureReviewPrompt
all_prompts = LiteratureReviewPrompt()


"""
Take the idea.json generated from the previous script and output a perfectly synthesized, exhaustive literature_review.md
"""
class SearchQueries(BaseModel):
    queries: List[str] = Field(..., description="5 distinct semantic scholar search queries (broad, specific, methodological).")

# Request openAccessPdf in the API call
SEMANTIC_SCHOLAR_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,venue,citationCount,openAccessPdf"
# --- NEW: API KEY CONFIGURATION ---
S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
# If you have an API key, we can allow 10 concurrent requests.
# If not, fallback to 1 to protect against 429.
max_concurrent = 5 if S2_API_KEY else 1
S2_SEMAPHORE = asyncio.Semaphore(max_concurrent)

# Load local embedding model (automatically uses GPU if available)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading embedding model on: {device}")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

memory = MemoryManager()

# ==========================================
# GRAPH TRAVERSAL & RETRIEVAL (I/O Bound)
# ==========================================
async def fetch_and_parse_pdf(session: aiohttp.ClientSession, url: str) -> str:
    """Downloads an Open Access PDF and extracts the Method/Conclusion sections."""
    if not url: return ""
    try:
        # 10-second timeout to prevent hanging on bad servers
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                return ""

            # Ensure it's actually a PDF and not an HTML landing page
            content_type = response.headers.get('Content-Type', '')
            if 'application/pdf' not in content_type:
                return ""

            pdf_bytes = await response.read()

            # Parse PDF in memory
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"

            doc.close()

            # Heuristic Compression: Get Intro/Methods (Front) and Conclusion (Back)
            # Skip the middle to save token context
            if len(full_text) < 6000:
                return full_text
            else:
                front = full_text[:3000]
                back = full_text[-3000:]
                return f"{front}\n\n...[PDF MIDDLE TRUNCATED]...\n\n{back}"
    except Exception as e:
        print(f"[PDF Fetch Failed] {url} - {str(e)[:50]}")
        return ""


async def enrich_papers_with_pdfs(papers: List[Dict]) -> List[Dict]:
    """Asynchronously fetches PDF text for all valid papers."""
    print(f"\n--- Fetching Full-Text PDFs for {len(papers)} papers ---")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for p in papers:
            # CHECK MEMORY FIRST
            cached_text = memory.load_cached_paper(p["paperId"])
            if cached_text:
                print(f"   [Memory Hit] Loaded {p['paperId']} from notes/")
                p["full_text"] = cached_text
            else:
                pdf_url = p.get("openAccessPdf", {}).get("url") if p.get("openAccessPdf") else None
                tasks.append((p, fetch_and_parse_pdf(session, pdf_url)))

        # Only fetch what wasn't cached
        for p, task in tasks:
            text = await task
            p["full_text"] = text
            if text:
                print(f"   ✓ Deep-read PDF for: {p['title'][:40]}...")
                # SAVE TO MEMORY
                memory.save_paper_extract(p["paperId"], p["title"], text)
    return papers


# ==========================================
# 3. GRAPH TRAVERSAL & FILTERING
# ==========================================
async def fetch_semantic_scholar(session: aiohttp.ClientSession, url: str, params: dict = None) -> dict:
    """Helper to fetch from API with API Key authentication and high concurrency."""
    max_retries = 5

    async with S2_SEMAPHORE:
        # Only sleep if we DON'T have an API key
        if not S2_API_KEY:
            await asyncio.sleep(1.0)

        for attempt in range(max_retries):
            try:
                # NEW: Inject the S2_HEADERS here
                async with session.get(url, params=params, headers=S2_HEADERS) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        wait_time = 2 ** (attempt+1)
                        print(f"   [API 429] Rate limited. Pausing for {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"   [Literature Scout] S2 API returned status {response.status}. Retrying...")
                        await asyncio.sleep(2 ** (attempt+1))
            except Exception as e:
                print(f"[Connection Error] {str(e)[:50]}")
                # exit()
                return {}
    return {}
    # exit()


async def expand_search_and_traverse(idea: Dict) -> List[Dict]:
    print("\n--- Phase: Search & Graph Traversal ---")

    # CLEAN UP QUERIES:
    # 1. Remove underscores from Name
    name_clean = idea.get('Name', '').replace('_', ' ')

    title = idea.get('Title', '')
    blob_title = TextBlob(title)
    noun_phrases = blob_title.noun_phrases
    meaningful_words = [word for word in noun_phrases]
    print(f"   {len(meaningful_words)} noun_phrases found in the title for paper searching.")

    # # 3. Method keyword
    # method_clean = idea.get('Simulations', '').split()[0] if idea.get('Simulations') else ""

    queries = [name_clean,]
    queries.extend(meaningful_words)

    # Remove duplicates or empty queries
    queries = list(set([q for q in queries if len(q) > 3]))

    print(f"Executing {len(queries)} base searches with queries: {queries}")

    all_papers = {}
    async with aiohttp.ClientSession() as session:
        # Step A: Base Search
        for q in queries:
            url = f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/search"
            # We ask for 10 limit now to ensure we get a good base pool
            res = await fetch_semantic_scholar(session, url, {"query": q, "limit": 10, "fields": FIELDS})

            for p in res.get("data", []):
                # Keep if it has an ID, even if abstract is null (we might fetch PDF later)
                if p.get("paperId"):
                    all_papers[p["paperId"]] = p

        print(f"Found {len(all_papers)} base papers. Starting Graph Traversal...")

        # Step B: Graph Traversal (Fetch papers that CITE our base papers)
        top_base = sorted(all_papers.values(), key=lambda x: x.get("citationCount", 0), reverse=True)[:5]

        tasks = []
        for base in top_base:
            url = f"{SEMANTIC_SCHOLAR_GRAPH_API}/paper/{base['paperId']}/citations"
            # Note: The citations endpoint uses slightly different field names
            tasks.append(fetch_semantic_scholar(session, url, {
                "fields": "citingPaper.paperId,citingPaper.title,citingPaper.abstract,citingPaper.year,citingPaper.openAccessPdf",
                "limit": 10}))

        results = await asyncio.gather(*tasks)
        for res in results:
            for citation in res.get("data", []):
                p = citation.get("citingPaper", {})
                if p and p.get("paperId"):
                    # Normalize the dictionary keys to match base papers
                    all_papers[p["paperId"]] = p

    print(f"Graph traversal complete. Total unique papers retrieved: {len(all_papers)}")
    return list(all_papers.values())


# ==========================================
# 3. SEMANTIC FILTERING (GPU/CPU Bound)
# ==========================================
def filter_top_k_papers(idea: Dict, papers: List[Dict], top_k: int = 20):
    """Uses local embeddings to rank papers by semantic similarity to our hypothesis."""
    print("\n--- Phase: Semantic Filtering ---")
    if not papers: return []
    hypothesis_text = idea["Title"] + " " + idea["Short Hypothesis"] + " " + idea["Simulations"]
    paper_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]

    print("Computing embeddings...")
    # Compute embeddings
    idea_emb = embedding_model.encode(hypothesis_text, convert_to_tensor=True)
    paper_embs = embedding_model.encode(paper_texts, convert_to_tensor=True)

    # Calculate Cosine Similarity
    cosine_scores = util.cos_sim(idea_emb, paper_embs)[0]

    # Sort and get top K
    top_results = torch.topk(cosine_scores, k=min(top_k, len(papers)))

    filtered_papers = []
    print(f"Top {top_k} Most Relevant Papers:")
    for score, idx in zip(top_results[0], top_results[1]):
        paper = papers[idx]
        print(f" - [Score: {score.item():.2f}] {paper.get('title', 'Unknown')}")
        filtered_papers.append(paper)
    return filtered_papers, paper_embs


# ==========================================
# 4. CITATION CHECKER
# ==========================================
def get_hallucinated_citations(review_text: str, valid_ids: Set[str]) -> Set[str]:
    """Scans text for [ID] formatting and checks if the ID is real."""
    # Matches exactly 40 lowercase hex characters inside brackets
    cited_ids = set(re.findall(r'\[([a-f0-9]{40})\]', review_text))
    hallucinated = cited_ids - valid_ids
    return hallucinated


# ==========================================
# 4. MAP-REDUCE & MULTI-PASS SYNTHESIS PIPELINE
# ==========================================
async def generate_mini_review(batch_idx: int, papers: List[Dict], idea: Dict, client, model) -> str:
    """MAP phase: Analyzes a small batch of papers."""
    print(f"Generating mini-review for batch {batch_idx}...")

    papers_text = ""
    for p in papers:
        papers_text += f"\n--- PAPER ID: [{p['paperId']}] ---\nTitle: {p['title']}\nAbstract: {p['abstract']}\n"
        if p.get("full_text"):
            papers_text += f"Deep PDF Extract (Methods/Conclusion):\n{p['full_text']}\n"

    prompt = all_prompts.mini_review_prompt(idea, papers_text)

    loop = asyncio.get_event_loop()
    # Note: If your LLM client supports temperature, set it to a low value here (e.g., 0.2) for factual recall.
    response, _ = await loop.run_in_executor(None, get_response_from_llm, prompt, client, model,
                                             "You are a scientific writer.", [])
    return response


async def generate_draft_synthesis(draft_idx: int, mini_reviews: List[str], idea: Dict, client, model) -> str:
    """Generates one of several distinct drafts."""
    print(f"Generating Synthesis Draft {draft_idx}...")
    combined_reviews = "\n\n".join(mini_reviews)

    prompt = all_prompts.draft_synthesis_prompt(idea, combined_reviews)

    loop = asyncio.get_event_loop()
    response, _ = await loop.run_in_executor(None, get_response_from_llm, prompt, client, model, "You are an elite PI.",
                                             [])
    return response


async def evaluate_and_combine_reviews(strategy_name: str, strategy_instruction: str, drafts: List[str], idea: Dict, valid_ids: Set[str], client, model) -> str:
    """THE EDITOR-IN-CHIEF: Evaluates drafts and combines them using a specific editorial strategy."""
    print(f"\n--- Phase: Generating Synthesis ({strategy_name}) ---")
    drafts_text = ""
    for i, draft in enumerate(drafts):
        drafts_text += f"\n\n=== DRAFT {i + 1} ===\n{draft}\n"

    prompt = all_prompts.all_combined_prompt(idea, drafts, drafts_text, strategy_name, strategy_instruction)

    loop = asyncio.get_event_loop()

    # Retry loop for strict citation checking
    max_retries = 2
    for attempt in range(max_retries):
        response, _ = await loop.run_in_executor(None, get_response_from_llm, prompt, client, model,
                                                 "You are a ruthlessly critical Editor-in-Chief.", [])

        # Verify Citations
        hallucinated = get_hallucinated_citations(response, valid_ids)
        if not hallucinated:
            print(f"   ✓ Citation Check Passed ({strategy_name})")
            return response
        else:
            print(
                f"   ! WARNING: Hallucinated citations in {strategy_name}. Forcing rewrite (Attempt {attempt + 1}/{max_retries}).")
            prompt += f"\n\n*** CORRECTION REQUIRED: You invented these fake citations: {hallucinated}. Rewrite the review and ONLY use the IDs provided in the drafts! ***"

        print(f"   ! WARNING: Max retries hit for {strategy_name}. Removing hallucinated citations manually.")
        for bad_id in hallucinated:
            response = response.replace(f"[{bad_id}]", "[Citation Removed]")
        return response


# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def cluster_and_batch_papers(papers: List[Dict], embeddings: torch.Tensor, num_clusters: int = 3):
    """Clusters papers by topic so each mini-review has a cohesive theme."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings.cpu().numpy())

    batches = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        batches[label].append(papers[idx])

    return list(batches.values())


async def run_exhaustive_review(idea_path: str, model_name, num_drafts, citation_num=20, main_target='Default'):
    client, client_model = create_client(model_name)

    with open(idea_path, "r", encoding="utf-8") as f:
        ideas = json.load(f)
        target_idea = ideas[0]
        if "idea" in target_idea:
            target_idea = target_idea["idea"]

    print(f"Target Idea: {target_idea['Title']}")

    # 1. Graph Traversal (Assume expand_search_and_traverse is defined above)
    raw_papers = await expand_search_and_traverse(target_idea)

    # 2. Semantic Filter (Assume filter_top_k_papers is defined above)
    cite_number = min(len(raw_papers), citation_num)
    top_papers, paper_embedding = filter_top_k_papers(target_idea, raw_papers, top_k=cite_number)

    enriched_papers = await enrich_papers_with_pdfs(top_papers)
    valid_paper_ids = {p["paperId"] for p in enriched_papers}

    # 3. MAP: Generate mini-reviews
    # batches = cluster_and_batch_papers(top_papers, paper_embedding)
    batch_size = 8
    batches = [enriched_papers[i:i + batch_size] for i in range(0, len(enriched_papers), batch_size)]
    map_tasks = [generate_mini_review(i + 1, b, target_idea, client, client_model) for i, b in enumerate(batches)]
    mini_reviews = await asyncio.gather(*map_tasks)

    # 4. REDUCE (Multi-Pass): Generate distinct drafts
    draft_tasks = [generate_draft_synthesis(i + 1, mini_reviews, target_idea, client, client_model) for i in
                   range(num_drafts)]
    drafts = await asyncio.gather(*draft_tasks)

    # 5. EVALUATE & COMBINE: Multiple Editorial Strategies
    all_strategies = all_prompts.strategy_focus()
    for key, value in all_strategies:
        if key.lower() == main_target.lower():
            strategies = [(key, value)]
            print("Find the strategy named ", main_target)
            break
    if not strategies:
        print("There is no strategy named ", main_target)
        print("Choose one of them: 'Default', 'Gap-Focused', 'Experiment', 'Model', 'Narrative'")

    editor_tasks = [
        evaluate_and_combine_reviews(name, instruction, drafts, target_idea, valid_paper_ids, client, client_model)
        for name, instruction in strategies
    ]

    final_reviews = await asyncio.gather(*editor_tasks)

    # Save output
    out_path = idea_path.replace(".json", "_literature_review.md")
    if os.path.exists(out_path):
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        backup_path = out_path.replace(".md", f"_{suffix}.md")
        os.rename(out_path, backup_path)
        print(f"Existing literature review file backed up as: {backup_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Exhaustive Literature Review\n\n**Idea:** {target_idea['Title']}\n\n")
        f.write(f"## INITIAL DRAFT:\n")
        for i in range(len(drafts)):
            f.write(f"### Draft {i + 1}\n")
            f.write(drafts[i])
            f.write("\n")

        f.write(f"\n\n\n========== COMBINED REVIEW ==========\n")
        # Write each variation to the markdown file
        for idx, (name, _) in enumerate(strategies):
            f.write(f"## Variation: {name}\n\n")
            f.write(final_reviews[idx])
            f.write("\n\n---\n\n")

        # Add a bibliography key at the bottom
        f.write("\n\n## References Key\n")
        for p in enriched_papers:
            f.write(f"- **[{p['paperId']}]**: {p['title']} ({p.get('year', 'N/A')})\n")

    print(f"\nSuccess! Saved review to {out_path}")


if __name__ == "__main__":
    # Ensure you point this to the JSON file generated by your `generate_idea.py`
    parser = argparse.ArgumentParser(description="Generate AI collaborator proposals")
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
        default=1,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--idea-file",
        type=str,
        default="generated_files/idea_oscillation.json",
        help="Path to the workshop description file.",
    )
    parser.add_argument(
        "--citation-num",
        type=int,
        default=16,
        help="Maximum number of citations per proposal.",
    )
    parser.add_argument(
        "--main_target",
        type=str,
        default='Default',
        choices=['Default', 'Gap-Focused', 'Experiment', 'Model', 'Narrative'],
        help="The main focus of the proposal.",
    )
    args = parser.parse_args()

    asyncio.run(run_exhaustive_review(args.idea_file, args.model, args.max_num_generations, args.citation_num, args.main_target))

