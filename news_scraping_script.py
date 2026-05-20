import csv
import glob
import logging
import os
import re
import time
import nltk
import requests
from bs4 import BeautifulSoup
from logging.handlers import RotatingFileHandler
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse

DISCOVERED_SITES_FILE = "discovered_sites.txt"
LOG_FILE = "news_scraping.log"

# ── Logging setup ─────────────────────────────────────────────────────────
def _configure_logging():
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

_configure_logging()
logger = logging.getLogger(__name__)

# ── Keyword Configuration ─────────────────────────────────────────────────────
# Articles not matching at least one keyword (or a morphological variant of it)
# are filtered out. Edit this list to focus the crawler on topics you care about.
KEYWORDS = [
    "election",
    "government",
    "economy",
    "security",
    "flood",
    "politics",
    "protest",
    "conflict",
    "health",
    "education",
]


class KeywordMatcher:
    """
    Matches article text against a keyword list using two complementary techniques:

    1. Regex (word-boundary) matching — exact forms with no false positives
       (e.g. 'Iran' will not match 'irritant').
    2. NLTK WordNet lemmatisation — morphological variant matching so that
       'election' also matches 'elections', 'elected', 'electing', etc.

    Both techniques operate on the full container text so summary snippets
    are included alongside the headline.
    """

    def __init__(self, keywords):
        self._ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.keywords = keywords
        # Pre-lemmatise keywords once so comparisons at match-time are fast
        self._keyword_lemmas = {
            kw: self.lemmatizer.lemmatize(kw.lower()) for kw in keywords
        }
        # One compiled regex per keyword — word boundaries prevent partial hits
        self._patterns = {
            kw: re.compile(r"\b" + re.escape(kw.lower()) + r"\b", re.IGNORECASE)
            for kw in keywords
        }

    @staticmethod
    def _ensure_nltk_data():
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

    def matched_keywords(self, text):
        """
        Returns the list of keywords found in *text*.
        An empty list means the article is not relevant.
        """
        if not text:
            return []

        text_lower = text.lower()
        # Tokenise with regex — no extra NLTK corpus download required
        token_lemmas = {
            self.lemmatizer.lemmatize(t)
            for t in re.findall(r"\b[a-z]+\b", text_lower)
        }

        matched = []
        for kw in self.keywords:
            # Technique 1: exact word-boundary regex match
            if self._patterns[kw].search(text_lower):
                matched.append(kw)
                continue
            # Technique 2: lemma overlap (plurals, verb inflections, etc.)
            if self._keyword_lemmas[kw] in token_lemmas:
                matched.append(kw)

        return matched


def load_discovered_sites(filepath=None):
    """
    Reads site URLs and metadata from a CSV file written by local_news_sites_finder.py.
    If filepath is None, the most recently modified *_news_sites.csv file in the
    working directory is used automatically.
    Returns a (config, metadata) tuple.
    """
    if filepath is None:
        matches = sorted(glob.glob("*_news_sites.csv"), key=os.path.getmtime, reverse=True)
        if not matches:
            logger.warning("No *_news_sites.csv file found. Run local_news_sites_finder.py first.")
            return {}, {}
        filepath = matches[0]
        logger.info(f"Using site list: '{filepath}'")

    if not os.path.exists(filepath):
        return {}, {}

    config = {}
    metadata = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            name = (row.get("name") or "").strip()

            # Skip blank rows and section-header rows (=== ... ===)
            if not url or name.startswith("==="):
                continue

            # Extract country/state from the first valid data row
            if not metadata.get("country") and row.get("country"):
                metadata["country"] = row["country"].strip()
            if not metadata.get("state") and row.get("state"):
                metadata["state"] = row["state"].strip()

            domain = urlparse(url).netloc.replace("www.", "")
            site_name = domain.split(".")[0] if domain else name
            config[site_name] = {
                "url": url,
                "container_selector": "article",
                "title_selector": "h2, h3, h1",
                "link_selector": "a",
            }

    logger.info(f"Loaded {len(config)} sites from '{filepath}'")
    return config, metadata

# Configuration for different news channels
# You will need to inspect the target sites to find the correct CSS selectors
NEWS_CONFIG = {
    "bbc": {
        "url": "https://www.bbc.com/news",
        "container_selector": "div.gs-c-promo",  # Example selector for article blocks
        "title_selector": "h3.gs-c-promo-heading__title",
        "link_selector": "a.gs-c-promo-heading",
    },
    "reuters": {
        "url": "https://www.reuters.com",
        "container_selector": "div.media-story-card__body__3g_cl", 
        "title_selector": "a[data-testid='Heading']",
        "link_selector": "a[data-testid='Heading']",
    }
}

class NewsCrawler:
    def __init__(self, config, keywords=None):
        self.config = config
        self.matcher = KeywordMatcher(keywords or KEYWORDS)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_page(self, url):
        """Fetches raw HTML from a URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def parse_news(self, site_name, html):
        """Parses the HTML based on the site's specific configuration mapping."""
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        site_meta = self.config[site_name]
        articles = []

        # Find all blocks containing articles
        containers = soup.select(site_meta["container_selector"])
        
        for container in containers:
            title_element = container.select_one(site_meta["title_selector"])
            link_element = container.select_one(site_meta["link_selector"])

            if title_element and link_element:
                title = title_element.get_text(strip=True)
                link = link_element.get("href")

                # Clean up relative URLs
                if link and link.startswith("/"):
                    base_url = "/".join(site_meta["url"].split("/")[:3])
                    link = base_url + link

                # Match against full container text for richer context
                container_text = container.get_text(separator=" ", strip=True)
                matched = self.matcher.matched_keywords(container_text)
                if not matched:
                    continue

                articles.append({
                    "source": site_name,
                    "title": title,
                    "link": link,
                    "matched_keywords": matched,
                })
        
        return articles

    def crawl_all(self):
        """Iterates through all configured sites and gathers articles."""
        all_results = []
        for site_name, target in self.config.items():
            logger.info(f"Crawling {site_name}...")
            html = self.fetch_page(target["url"])
            site_articles = self.parse_news(site_name, html)
            all_results.extend(site_articles)
            logger.info(f"Found {len(site_articles)} articles from {site_name}.")
            # Politeness delay: wait 2 seconds between hitting different sites
            time.sleep(2) 
            
        return all_results

# Execution
if __name__ == "__main__":
    # Merge hardcoded config with any sites discovered by local_news_sites_finder.py.
    # Hardcoded entries take priority (discovered sites won't overwrite them).
    discovered, site_metadata = load_discovered_sites()
    config = {**discovered, **NEWS_CONFIG}

    # Add the state name to keywords so articles mentioning the local state
    # are captured even if they don't match the base keyword list.
    active_keywords = list(KEYWORDS)
    state = site_metadata.get("state", "").strip()
    if state and state not in active_keywords:
        active_keywords.append(state)
        logger.info(f"Added '{state}' to keyword list for local relevance matching.")

    crawler = NewsCrawler(config, keywords=active_keywords)
    scraped_data = crawler.crawl_all()

    logger.info("--- Scraped Headlines ---")
    for idx, article in enumerate(scraped_data, 1):
        logger.info(f"{idx}. [{article['source'].upper()}] {article['title']}")
        logger.info(f"   Keywords : {', '.join(article['matched_keywords'])}")
        logger.info(f"   Link     : {article['link']}")