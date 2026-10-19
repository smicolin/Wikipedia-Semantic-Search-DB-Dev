import wikipediaapi
import re
import json
from pathlib import Path

### Goal: we need to grab an article and then clean its text, making sure not to bring in sources or other things that can dilute our semantic meaning

USER_AGENT = 'Wikipedia Semantic Search (colin.smith11@icloud.com)'
wiki = wikipediaapi.Wikipedia(USER_AGENT, language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

SKIP_SECTIONS = [
    "see also",
    "explanatory notes",
    "citations",
    "general and cited sources",
    "further reading"
]

class StopExtraction(Exception):
    """Signal to immediately stop parsing subsequent sections."""
    pass

def get_clean_page_text(page: wikipediaapi.WikipediaPage, ) -> str:
    """Get all of the text besides extra content like sources and further reading"""
    paragraphs = []

    if page.summary:
        paragraphs.append(page.summary+'\n')

    def filter_sections(sections):
        nonlocal paragraphs
        for section in sections:
            title_clean = section.title.strip().lower()

            if title_clean in SKIP_SECTIONS:
                raise StopExtraction()
            elif section.text.strip():
                paragraphs.append('##'+section.title.strip()+'\n')
                paragraphs.append(section.text+'\n')

            filter_sections(section.sections) # Using recursion to process subsections

    try:
        filter_sections(page.sections)
    except StopExtraction:
        pass

    paragraphs_as_string = "\n".join(paragraphs)
    return re.sub(r'\s{4,}(?!\n##)', ' ',paragraphs_as_string)

def get_articles_from_category(category_name: str) -> list[str]:
    """Retrieves article titles from a given Wikipedia category."""
    cat_page = wiki.page(f"Category:{category_name}")
    if not cat_page.exists():
        print(f"Category '{category_name}' not found.")
        return []

    titles = []
    
    for member in cat_page.categorymembers.values():
        if member.ns == wikipediaapi.Namespace.MAIN:
            titles.append(member.title)
        
    return titles

### Categories with Math/ML focus that we want to store in database
TARGET_CATEGORIES = [
    "Linear_algebra",
    "Artificial intelligence",
    "Machine_learning",
    "Probability_theory",
    "Deep_learning",
    "Statistical_theory",
    "Mathematical_optimization",
    "Natural_language_processing",
    "Reinforcement_learning",
    "Artificial_neural_networks"
]

def gather_all_titles(categories: list[str]) -> list[str]:
    """Looping through Wikipedia target categories to get article titles"""
    print("Gathering article titles across categories...")
    all_titles = []
    for category in categories:
        titles = get_articles_from_category(category)
        print(f"  • Category '{category}': found {len(titles)} articles")
        all_titles.extend(titles)

    # Deduplicate titles
    unique_titles = list(dict.fromkeys(all_titles))
    print(f"{len(unique_titles)} unique articles collected.")
    return unique_titles

def run_ingestion(titles: list[str]) -> list[dict]:
    """Using list of all article titles to pull in text of articles"""
    articles = []
    total_titles = len(titles)

    print(f"\nStarting ingestion for {total_titles} articles...")

    for idx, title in enumerate(titles, 1):
        try:
            page = wiki.page(title)
            if page.exists():
                clean_text = get_clean_page_text(page)
                if clean_text.strip():
                    articles.append({
                        "id": idx,
                        "title": page.title,
                        "url": page.fullurl,
                        "text": clean_text
                    })
        except Exception as e:
            print(f"Skipped '{title}' due to error: {e}")

        # Print download progress for every 100 titles
        if idx % 100 == 0 or idx == total_titles:
            print(f"  • Processed {idx}/{total_titles} articles ({len(articles)} saved)...")
    return articles

def save_dataset(articles: list[dict]) -> None:
    """Save article text as JSON"""
    output_file = Path(__file__).resolve().parent.parent / "data/all_articles.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\nIngestion complete Successfully saved {len(articles)} cleaned articles to '{output_file}'.")

if __name__ == "__main__":
    titles = gather_all_titles(TARGET_CATEGORIES)
    dataset = run_ingestion(titles)
    save_dataset(dataset)