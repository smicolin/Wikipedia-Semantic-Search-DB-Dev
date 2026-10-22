import json
import re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Chunking
def create_chunks_from_article(article: dict, target_words: int = 250, overlap_words: int = 30) -> list[dict]:
    """Splits article text into chunks, tracking '##' section headers for context injection."""
    article_id = article["id"]
    title = article["title"]
    url = article["url"]
    text = article["text"]

    # Split text by double newlines into blocks
    raw_blocks = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_words = []
    current_section = "Introduction"
    chunk_counter = 0

    for block in raw_blocks:
        # Detect if this block is a section header (e.g. "## History" or "##Vector spaces")
        if block.startswith("##"):
            # Extract section title without '##'
            current_section = re.sub(r"^##\s*", "", block).strip()
            continue

        # Split regular body paragraph into words
        block_words = block.split()
        current_words.extend(block_words)

        # Generate chunk when word limit is reached
        while len(current_words) >= target_words:
            chunk_word_list = current_words[:target_words]
            raw_chunk_text = " ".join(chunk_word_list)

            # Context Header Injection with Section Title
            contextual_text = (
                f"Article: {title} | Section: {current_section}\n"
                f"{raw_chunk_text}"
            )

            chunks.append(
                {
                    "chunk_id": f"art_{article_id}_chunk_{chunk_counter}",
                    "article_id": article_id,
                    "title": title,
                    "section": current_section,
                    "url": url,
                    "text": contextual_text,  # For vector embeddings
                    "raw_text": raw_chunk_text,  # Clean snippet display
                }
            )

            chunk_counter += 1
            current_words = current_words[target_words - overlap_words :]

    # Handle final remaining words
    if current_words:
        raw_chunk_text = " ".join(current_words)
        contextual_text = (
            f"Article: {title} | Section: {current_section}\n{raw_chunk_text}"
        )

        chunks.append(
            {
                "chunk_id": f"art_{article_id}_chunk_{chunk_counter}",
                "article_id": article_id,
                "title": title,
                "section": current_section,
                "url": url,
                "text": contextual_text,
                "raw_text": raw_chunk_text,
            }
        )

    return chunks


def process_all_articles(input_path: Path, output_path: Path) -> list[dict]:
    """Loads all raw articles from JSON, chunks them, and saves the chunk dataset."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find '{input_path}'. Run ingestion first!"
        )

    print(f"Loading raw articles from '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_chunks = []
    print(f"Chunking {len(articles)} articles...")

    for article in articles:
        article_chunks = create_chunks_from_article(article)
        all_chunks.extend(article_chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(
        f"Done! Generated {len(all_chunks)} total chunks from {len(articles)} articles."
    )
    print(f"Saved chunk dataset to: '{output_path}'\n")

    return all_chunks

# Embedding

def generate_embeddings(
    chunks: list[dict],
    output_vectors_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
):
    """Takes text chunks, converts them into dense vector embeddings,

    normalizes them for fast cosine similarity, and saves the matrix to disk.
    """
    texts_to_embed = [chunk["text"] for chunk in chunks]
    total_chunks = len(texts_to_embed)

    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(
        f"Generating embeddings for {total_chunks} chunks (this may take a minute)..."
    )
    embeddings = model.encode(
        texts_to_embed,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # L2 Normalization (Crucial Step!)
    # Rescales every vector so its length equals 1, allowing cosine similarity via dot products.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = embeddings / norms

    # Save 2D NumPy array matrix to disk
    output_vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_vectors_path, normalized_embeddings)

    print("\nEmbedding process complete!")
    print(f"Matrix Shape: {normalized_embeddings.shape}")
    print(f"  • {normalized_embeddings.shape[0]} total vectors (chunks)")
    print(f"  • {normalized_embeddings.shape[1]} dimensions per vector")
    print(f"Saved vectors matrix to: '{output_vectors_path}'")

if __name__ == "__main__":
    # Standardize data folder path relative to script root
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

    ARTICLES_FILE = DATA_DIR / "all_articles.json"
    CHUNKS_FILE = DATA_DIR / "all_chunks.json"
    EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"

    # Step 1: Chunk raw articles
    chunks = process_all_articles(
        input_path=ARTICLES_FILE, output_path=CHUNKS_FILE
    )

    # Step 2: Generate vector embeddings from chunks
    generate_embeddings(chunks=chunks, output_vectors_path=EMBEDDINGS_FILE)