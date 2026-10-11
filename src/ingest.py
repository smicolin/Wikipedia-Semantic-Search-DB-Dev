import wikipediaapi
import re

### Goal: we need to grab an article and then clean its text, making sure not to bring in sources or other things that can dilute our semantic meaning

USER_AGENT = 'Wikipedia Semantic Search (redact)'
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

page1 = wiki.page('Linear_algebra')

print(get_clean_page_text(page1))