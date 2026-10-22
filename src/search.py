import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_cli_snippet(text: str) -> str:
    """Strips raw LaTeX markup, removes extra whitespace, and prepares text for CLI display."""
    # Remove LaTeX math code blocks: {\displaystyle ...}
    text = re.sub(r"\{\\displaystyle\s*[^}]*\}", "", text)
    # Remove remaining bare LaTeX commands (e.g. \displaystyle)
    text = re.sub(r"\\\w+(\{[^}]*\})*", "", text)
    # Clean up formatting artifact braces/symbols left behind
    text = re.sub(r"[{}]", "", text)
    # Collapse multiple spaces and newlines into single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SemanticSearchEngine:

    def __init__(
        self,
        chunks_filename: str = "all_chunks.json",
        vectors_filename: str = "embeddings.npy",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        data_dir = Path(__file__).resolve().parent.parent / "data"
        chunks_path = data_dir / chunks_filename
        vectors_path = data_dir / vectors_filename

        if not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                "Search index files missing. Run embed.py first."
            )

        print("Loading search index into memory...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.embeddings = np.load(vectors_path)

        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print("Search Engine ready!\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Encodes user query, computes cosine similarity via dot product,"""
        query_vector = self.model.encode(query, convert_to_numpy=True)

        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Matrix dot product across all chunk vectors
        scores = np.dot(self.embeddings, query_vector)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk_data = self.chunks[idx].copy()
            chunk_data["score"] = float(scores[idx])
            chunk_data["clean_snippet"] = clean_cli_snippet(
                chunk_data["raw_text"]
            )
            results.append(chunk_data)

        return results


def print_results(matches: list[dict]):
    """Formatted CLI display function."""
    print("\n" + "=" * 70)
    for rank, match in enumerate(matches, 1):
        print(f"[{rank}] {match['title']}  (Score: {match['score']:.4f})")
        print(f"    URL:  {match['url']}")
        print(f"    Text: {match['clean_snippet'][:350]}...")
        print("-" * 70)
    print("\n")


# Interactive CLI Main Loop
if __name__ == "__main__":
    engine = SemanticSearchEngine()

    print("=" * 70)
    print("      WIKIPEDIA SEMANTIC SEARCH INTERACTIVE CLI")
    print("      Type your query below or type 'exit' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_query = input("Search query > ").strip()

            if not user_query:
                continue

            if user_query.lower() in ["exit"]:
                print("Exiting search engine.")
                break

            matches = engine.search(user_query, top_k=3)
            print_results(matches)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting search engine.")
            break