# Wikipedia Semantic Article Search (In Progress)

### Purpose

Keeping up with the newest trends in ML and AI can be exhausting. That's why I created this search engine.
If you were to search "What is the fundamental math behind LLMs?" on Wikipedia, you would get a large quantity of irrelevant articles that do not answer your question.
Using this custom search engine, the most relevant articles can easily be found.

### How does it work?

Raw article text is extracted from the Wikipedia API and cleaned. These articles are broken up into chunks and passed into an embedding model that stores data in a 384-dimensional vector space.

When a user asks a question, their question is processed by the embedding model and the mathematical similarity of the question to the articles in the database is compared, with the highest matches being presented to the user.

Happy searching!