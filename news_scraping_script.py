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
ARTICLES_FILE = "scraped_articles.txt"

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

# ── Site Legitimacy Scoring ──────────────────────────────────────────────────
# Fully dynamic — no hardcoded outlet names.  Uses the same signals as
# local_news_sites_finder.py so CSV filtering is consistent.

_WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

_SOCIAL_PLATFORMS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "blogspot", "wordpress.com",
    "wix.com", "weebly.com", "tumblr.com",
}
_SPAM_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".xyz", ".click", ".top", ".win", ".buzz", ".loan",
}


def _get_domain_age_years(domain):
    """Returns approximate domain age in years via the Wayback Machine CDX API."""
    try:
        resp = requests.get(
            _WAYBACK_CDX,
            params={
                "url": domain,
                "matchType": "domain",
                "output": "json",
                "limit": "1",
                "fl": "timestamp",
            },
            timeout=8,
        )
        data = resp.json()
        if len(data) > 1:
            year = int(data[1][0][:4])
            return time.gmtime().tm_year - year
    except Exception:
        pass
    return 0


def _score_site(site):
    """
    Dynamic legitimacy score (0–100). No hardcoded domain lists.
    Signals: social-platform disqualifier, HTTPS, TLD quality, Wayback domain age.
    """
    url = site.get("url", "")
    domain = url.split("//")[-1].split("/")[0].replace("www.", "").lower()

    if any(p in domain for p in _SOCIAL_PLATFORMS):
        return 0

    score = 25
    if url.startswith("https://"):
        score += 15
    if re.search(r"\.gov(\.[a-z]{2,3})?$|\.edu(\.[a-z]{2,3})?$", domain):
        score += 20
    elif any(domain.endswith(t) for t in _SPAM_TLDS):
        score -= 20

    age = _get_domain_age_years(domain)
    if age >= 10:
        score += 30
    elif age >= 5:
        score += 20
    elif age >= 2:
        score += 10

    return max(0, min(score, 100))


# ── Keyword Configuration ─────────────────────────────────────────────────────
# Articles not matching at least one keyword (or a morphological variant of it)
# are filtered out. Edit this list to focus the crawler on topics you care about.
KEYWORDS = [
    "government",
    "construction",
    "security",
    "politics",
    "protest",
    "accident",
    "riots",
    "terrorist attack",
    "military deployment",
    "police deployment"
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
        # Each keyword may contain multiple words.  A list of per-word patterns
        # is stored so that ALL words must appear in the text (anywhere, not
        # necessarily adjacent).  This lets compound keywords like
        # "security Plateau" match "security incident in Plateau State".
        self._patterns = {
            kw: [
                re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
                for w in kw.lower().split()
            ]
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
            # Technique 1: all constituent words must appear in the text
            # (individually, not necessarily adjacent).
            if all(p.search(text_lower) for p in self._patterns[kw]):
                matched.append(kw)
                continue
            # Technique 2: lemma fallback for single-word keywords only
            if len(self._patterns[kw]) == 1 and self._keyword_lemmas[kw] in token_lemmas:
                matched.append(kw)

        return matched


def load_discovered_sites(filepath=None, top_n_national=10, top_n_state=5):
    """
    Reads site URLs and metadata from a CSV file written by local_news_sites_finder.py.
    If filepath is None, the most recently modified *_news_sites_updated.csv file in the
    working directory is used automatically.
    Sites are scored dynamically (Wayback domain age + TLD/HTTPS signals) and
    the top *top_n_national* national plus up to *top_n_state* state sites are
    returned.  Returns a (config, metadata) tuple.
    """
    if filepath is None:
        matches = sorted(glob.glob("*_news_sites_updated.csv"), key=os.path.getmtime, reverse=True)
        if not matches:
            logger.warning("No *_news_sites_updated.csv file found. Run local_news_sites_finder.py first.")
            return {}, {}
        filepath = matches[0]
        logger.info(f"Using site list: '{filepath}'")

    if not os.path.exists(filepath):
        return {}, {}

    national_rows = []
    state_rows = []
    metadata = {}
    current_section = "national"

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            name = (row.get("name") or "").strip()

            # Detect section-header rows and track which section we are in
            if name.startswith("==="):
                if "STATE" in name or "LOCAL" in name:
                    current_section = "state"
                else:
                    current_section = "national"
                continue

            if not url:
                continue

            # Extract country/state from the first valid data row
            if not metadata.get("country") and row.get("country"):
                metadata["country"] = row["country"].strip()
            if not metadata.get("state") and row.get("state"):
                metadata["state"] = row["state"].strip()

            entry = {"url": url, "name": name, "score": _score_site({"url": url, "name": name})}
            if current_section == "state":
                state_rows.append(entry)
            else:
                national_rows.append(entry)

    # Rank and select top sites from each section
    top_national = sorted(national_rows, key=lambda e: e["score"], reverse=True)[:top_n_national]
    top_state = (
        state_rows
        if len(state_rows) <= top_n_state
        else sorted(state_rows, key=lambda e: e["score"], reverse=True)[:top_n_state]
    )

    logger.info(
        f"Selected {len(top_national)} national + {len(top_state)} state sites "
        f"from '{filepath}' ({len(national_rows)} national + {len(state_rows)} state total)."
    )

    config = {}
    for entry in top_national + top_state:
        domain = urlparse(entry["url"]).netloc.replace("www.", "")
        site_name = domain.split(".")[0] if domain else entry["name"]
        config[site_name] = {
            "url": entry["url"],
            "container_selector": "article",
            "title_selector": "h2, h3, h1",
            "link_selector": "a",
        }

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

    def fetch_article_content(self, url):
        """Fetches and extracts the main text content from an article page."""
        html = self.fetch_page(url)
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        # Try increasingly broad selectors until we find substantial text
        for selector in ("article", "main", "[class*='article']", "[class*='content']", "body"):
            tag = soup.select_one(selector)
            if tag:
                for unwanted in tag(["script", "style", "nav", "header", "footer", "aside"]):
                    unwanted.decompose()
                text = tag.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text
        return ""

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

            # Follow each article link to fetch its full text content
            for article in site_articles:
                link = article.get("link")
                if link:
                    logger.info(f"  Fetching content: {article['title'][:70]}...")
                    article["content"] = self.fetch_article_content(link)
                    time.sleep(1)  # Politeness delay between article fetches

            all_results.extend(site_articles)
            logger.info(f"Found {len(site_articles)} articles from {site_name}.")
            # Politeness delay between sites
            time.sleep(2)

        return all_results

def save_articles_to_file(results, filepath=ARTICLES_FILE):
    """
    Writes all scraped articles to a plain text file, grouped by news source.
    Each article includes its headline, URL, matched keywords, and full content.
    """
    by_source = {}
    for article in results:
        by_source.setdefault(article["source"], []).append(article)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("SCRAPED NEWS ARTICLES\n")
        f.write(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total     : {len(results)} articles from {len(by_source)} sources\n")
        f.write("=" * 80 + "\n\n")

        for source, articles in by_source.items():
            f.write("=" * 80 + "\n")
            f.write(f"SOURCE: {source.upper()}\n")
            f.write("=" * 80 + "\n\n")

            for idx, article in enumerate(articles, 1):
                f.write(f"ARTICLE {idx}: {article['title']}\n")
                f.write(f"URL      : {article.get('link', 'N/A')}\n")
                f.write(f"KEYWORDS : {', '.join(article.get('matched_keywords', []))}\n")
                f.write("-" * 40 + "\n")
                content = article.get("content", "").strip()
                f.write(content if content else "[Content could not be retrieved]")
                f.write("\n\n")

    logger.info(f"Saved {len(results)} articles from {len(by_source)} sources to '{filepath}'")


# Execution
if __name__ == "__main__":
    # Merge hardcoded config with any sites discovered by local_news_sites_finder.py.
    # Hardcoded entries take priority (discovered sites won't overwrite them).
    discovered, site_metadata = load_discovered_sites()
    config = {**discovered, **NEWS_CONFIG}

    # Add the state name to keywords so articles mentioning the local state
    # are captured even if they don't match the base keyword list.
    state = site_metadata.get("state", "").strip()
    if state:
        active_keywords = [f"{kw} {state}" for kw in KEYWORDS]
        logger.info(f"Keywords scoped to state '{state}': {active_keywords}")
    else:
        active_keywords = list(KEYWORDS)

    crawler = NewsCrawler(config, keywords=active_keywords)
    scraped_data = crawler.crawl_all()

    logger.info("--- Scraped Headlines ---")
    for idx, article in enumerate(scraped_data, 1):
        logger.info(f"{idx}. [{article['source'].upper()}] {article['title']}")
        logger.info(f"   Keywords : {', '.join(article['matched_keywords'])}")
        logger.info(f"   Link     : {article['link']}")

    save_articles_to_file(scraped_data)