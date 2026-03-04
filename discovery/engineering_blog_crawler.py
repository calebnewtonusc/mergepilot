"""
Engineering Blog Crawler — Code review best practices from top engineering blogs

Crawls posts from:
  - Google Engineering Practices
  - Airbnb, Netflix, Stripe, Shopify, Meta, LinkedIn, etc.

Output: JSONL with article text, key review principles extracted
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger


BLOG_SOURCES = [
    {"name": "google_eng_practices", "url": "https://google.github.io/eng-practices/review/"},
    {"name": "airbnb_engineering", "url": "https://medium.com/airbnb-engineering"},
    {"name": "netflix_tech_blog", "url": "https://netflixtechblog.com/tagged/code-review"},
    {"name": "stripe_engineering", "url": "https://stripe.com/blog/engineering"},
    {"name": "shopify_engineering", "url": "https://shopify.engineering"},
]


@dataclass
class BlogPost:
    """A single engineering blog post about code review."""
    source: str
    title: str
    url: str
    content: str
    key_principles: list[str]


class EngineeringBlogCrawler:
    """Crawls engineering blog posts about code review best practices."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session = None

    async def crawl_all(self) -> None:
        """Crawl all configured blog sources."""
        output_file = self.output_dir / "engineering_blogs.jsonl"
        logger.info("Crawling engineering blogs...")

        async with aiohttp.ClientSession(
            headers={"User-Agent": "MergePilot Research / calebnewtonusc"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as session:
            self._session = session
            posts = []

            for source in BLOG_SOURCES:
                source_posts = await self._crawl_source(source)
                posts.extend(source_posts)
                logger.info(f"{source['name']}: {len(source_posts)} posts")

        async with aiofiles.open(output_file, "w") as f:
            for post in posts:
                await f.write(json.dumps(asdict(post)) + "\n")

        logger.info(f"Engineering blogs: {len(posts)} posts saved")

    async def _crawl_source(self, source: dict) -> list[BlogPost]:
        """Crawl a single blog source."""
        html = await self._fetch(source["url"])
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        posts = []

        # Extract main content
        content_div = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        if content_div:
            content = content_div.get_text()[:5000]
            principles = self._extract_principles(content)
            posts.append(BlogPost(
                source=source["name"],
                title=source["name"].replace("_", " ").title(),
                url=source["url"],
                content=content,
                key_principles=principles,
            ))

        return posts

    def _extract_principles(self, content: str) -> list[str]:
        """Extract actionable review principles from article text."""
        import re
        principles = []
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if len(line) > 40 and len(line) < 200:
                if any(kw in line.lower() for kw in ["should", "must", "always", "never", "prefer"]):
                    principles.append(line)
        return principles[:10]

    async def _fetch(self, url: str) -> Optional[str]:
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
        except Exception as e:
            logger.debug(f"Fetch failed {url}: {e}")
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    crawler = EngineeringBlogCrawler(output_dir="data/raw/blogs")
    asyncio.run(crawler.crawl_all())
