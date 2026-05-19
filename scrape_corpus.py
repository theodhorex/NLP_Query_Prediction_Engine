import requests
from bs4 import BeautifulSoup
import os
import re

class WikipediaCorpusScraper:
    def __init__(self, corpus_folder):
        self.corpus_folder = corpus_folder
        os.makedirs(corpus_folder, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_wikipedia_article(self, title):
        """Fetch article content dari Wikipedia"""
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {title}: {e}")
            return None

    def extract_text_from_html(self, html):
        """Extract plain text dari HTML"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove references like [1], [2]

        return text

    def save_article(self, title, content):
        """Save article content ke file"""
        filename = title.replace(' ', '_') + '.txt'
        filepath = os.path.join(self.corpus_folder, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Saved: {filename} ({len(content)} chars)")
            return True
        except Exception as e:
            print(f"✗ Error saving {filename}: {e}")
            return False

    def scrape_articles(self, article_titles):
        """Scrape multiple articles"""
        print(f"Scraping {len(article_titles)} articles from Wikipedia...")
        print("-" * 60)

        success_count = 0
        for title in article_titles:
            print(f"Fetching: {title}...", end=" ")
            html = self.get_wikipedia_article(title)

            if html:
                text = self.extract_text_from_html(html)
                if len(text) > 500:  # Minimal 500 characters
                    if self.save_article(title, text):
                        success_count += 1
                else:
                    print(f"✗ Too short ({len(text)} chars)")
            else:
                print("✗ Failed")

        print("-" * 60)
        print(f"Successfully scraped: {success_count}/{len(article_titles)} articles")
        return success_count

if __name__ == '__main__':
    corpus_folder = os.path.join(os.path.dirname(__file__), 'corpus')

    scraper = WikipediaCorpusScraper(corpus_folder)

    # List of Wikipedia articles untuk corpus
    articles = [
        "Machine learning",
        "Artificial intelligence",
        "Natural language processing",
        "Python (programming language)",
        "Computer science",
        "Data science",
        "Deep learning",
        "Neural network",
        "Information retrieval",
        "Search engine",
        "Text mining",
        "Computational linguistics",
        "Algorithm",
        "Database",
        "Big data",
        "Information extraction",
        "Word embedding",
        "Sentiment analysis",
        "Named entity recognition",
        "Machine translation",
        "Speech recognition",
        "Computer vision",
        "Supervised learning",
        "Unsupervised learning",
        "Reinforcement learning",
        "Feature extraction",
        "Classification (machine learning)",
        "Clustering",
        "Statistical model",
        "Probability theory"
    ]

    scraper.scrape_articles(articles)
