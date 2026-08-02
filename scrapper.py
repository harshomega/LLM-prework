import os
import re
import sys
import time
import hashlib

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

DATA_DIR = "data"
PAGE_LOAD_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 2

TAGS_TO_STRIP = ["script", "style", "nav", "footer", "header", "aside", "noscript"]


def build_driver() -> webdriver.Chrome:
    """Configure a headless Chrome driver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def extract_clean_text(html: str) -> str:
    """Strip boilerplate tags and return readable text, collapsing whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in TAGS_TO_STRIP:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator="\n")
    # collapse 3+ blank lines down to a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def url_to_filename(url: str) -> str:
    """Turn a URL into a safe, unique filename."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:80]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]  # avoid collisions on similar slugs
    return f"{slug}_{url_hash}.txt"


def scrape_urls(urls: list[str]):
    os.makedirs(DATA_DIR, exist_ok=True)
    driver = build_driver()
    saved_count = 0

    try:
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] Scraping {url} ...")
            try:
                driver.get(url)

                WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                html = driver.page_source
                text = extract_clean_text(html)

                if len(text) < 100:
                    print(f"  [WARN] Extracted very little text ({len(text)} chars) -- "
                          f"page may need a longer wait or a different extraction selector.")

                filename = url_to_filename(url)
                filepath = os.path.join(DATA_DIR, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"Source URL: {url}\n\n{text}")

                print(f"  Saved -> {filepath} ({len(text)} chars)")
                saved_count += 1

            except Exception as e:
                print(f"  [ERROR] Failed to scrape {url}: {e}")

            time.sleep(DELAY_BETWEEN_REQUESTS)

    finally:
        driver.quit()

    print(f"\nDone. Saved {saved_count}/{len(urls)} pages to ./{DATA_DIR}/")
    print("Next step: run 'python ingest.py' to index the new content.")


def load_urls_from_file(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scrape.py urls.txt")
        print("   or: python scrape.py <url1> <url2> ...")
        sys.exit(1)

    args = sys.argv[1:]


    if len(args) == 1 and os.path.isfile(args[0]):
        url_list = load_urls_from_file(args[0])
    else:
        url_list = args

    scrape_urls(url_list)