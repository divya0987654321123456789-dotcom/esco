import asyncio
import json
import os
import re
from urllib.parse import urljoin, urldefrag, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

START_URL = "https://www.ikioenergy.com/"
OUTPUT_DIR = "scrape_output"
USER_AGENT = "ikio-scraper-bot/1.0"
MAX_PAGES = 100
DELAY_BETWEEN = 1.0
TIMEOUT = 60  # seconds


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_url(url, base):
    """Resolve relative URL and strip fragments."""
    url = urljoin(base, url)
    url, _ = urldefrag(url)
    return url


def same_domain(url, root):
    return urlparse(url).netloc == urlparse(root).netloc


def parse_html(html, base_url):
    """Extract full page content (paragraphs, lists, headings, tables, etc.)."""
    soup = BeautifulSoup(html, "html.parser")

    # remove scripts and non-content elements
    for s in soup(["script", "style", "noscript", "iframe"]):
        s.decompose()

    title = soup.title.string.strip() if soup.title else ""
    m = soup.find("meta", attrs={"name": "description"})
    meta_desc = m["content"].strip() if m and m.get("content") else ""

    # Headings
    headings = {}
    for h in ["h1", "h2", "h3", "h4"]:
        headings[h] = [t.get_text(strip=True) for t in soup.find_all(h)]

    # Paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]

    # Lists
    lists = []
    for ul in soup.find_all(["ul", "ol"]):
        items = [li.get_text(strip=True) for li in ul.find_all("li")]
        if items:
            lists.append(items)

    # Tables
    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)

    # Images
    images = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            images.append({
                "src": urljoin(base_url, src),
                "alt": img.get("alt", "").strip()
            })

    # Links
    anchors = [normalize_url(a["href"], base_url)
               for a in soup.find_all("a", href=True)]

    # Combined visible text (fallback)
    full_text = soup.get_text(separator="\n", strip=True)

    return {
        "url": base_url,
        "title": title,
        "meta_description": meta_desc,
        "headings": headings,
        "paragraphs": paragraphs,
        "lists": lists,
        "tables": tables,
        "images": images,
        "links": anchors,
        "full_text": full_text
    }


async def safe_goto(page, url):
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT * 1000)
    except Exception as e:
        print(f"[warn] retrying {url} due to: {e}")
        try:
            await page.goto(url, wait_until="load", timeout=TIMEOUT * 1000)
        except Exception as e2:
            print(f"[error] failed second attempt for {url}: {e2}")
            return False
    return True


async def crawl():
    ensure_dir(OUTPUT_DIR)
    visited = set()
    to_visit = [START_URL]
    site_data = {}
    domain = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(START_URL))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()

        while to_visit and len(visited) < MAX_PAGES:
            url = to_visit.pop(0)
            if url in visited:
                continue

            try:
                print(f"[fetch] {url}")
                ok = await safe_goto(page, url)
                if not ok:
                    continue

                html = await page.content()
                data = parse_html(html, url)
                site_data[url] = data
                visited.add(url)

                # Save full structured JSON per page
                filename = re.sub(r"[^a-zA-Z0-9]+", "_", urlparse(url).path) or "home"
                with open(os.path.join(OUTPUT_DIR, f"{filename}.json"), "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Queue new same-domain links
                for link in data["links"]:
                    if same_domain(link, domain) and link not in visited and link not in to_visit:
                        to_visit.append(link)

                await asyncio.sleep(DELAY_BETWEEN)

            except Exception as e:
                print(f"[error] {url} -> {e}")

        await browser.close()

    # Save all pages together
    with open(os.path.join(OUTPUT_DIR, "site_data.json"), "w", encoding="utf-8") as f:
        json.dump(site_data, f, indent=2, ensure_ascii=False)

    print(f"[done] Scraped {len(visited)} pages. Output -> {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(crawl())
