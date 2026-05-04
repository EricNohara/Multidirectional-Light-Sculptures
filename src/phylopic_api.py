import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin


import requests


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass(frozen=True)
class PhyloPicImage:
    uuid: str
    title: str
    page_url: str
    preview_url: str
    download_url: str
    license_url: str = ""
    contributor: str = ""


class PhyloPicClient:
    def __init__(self, base_url="https://api.phylopic.org", timeout=15):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._build = None

    def search_images(self, query: str, limit: int = 12) -> list[PhyloPicImage]:
        query = query.strip()
        if not query:
            return []

        uuid = self.extract_uuid(query)
        if uuid:
            return [self.get_image(uuid)]

        node_uuid = self._find_best_node_uuid(query)
        if not node_uuid:
            return []

        data = self._get(
            "images",
            {
                "build": self.build,
                "page": 0,
                "embed_items": "true",
                "filter_clade": node_uuid,
            },
        )

        images = data.get("_embedded", {}).get("items", [])
        return [self._parse_image(item) for item in images[:limit]]

    def suggest_names(self, query: str, limit: int = 8) -> list[str]:

        query = query.strip()

        if not query or self.extract_uuid(query) or len(query) <= 1:
            return []

        data = self._get(
            "autocomplete",
            {
                "build": self.build,
                "query": query,
            },
        )
        matches = data.get("matches", [])


        return [match for match in matches[:limit] if isinstance(match, str)]

    def get_image(self, uuid: str) -> PhyloPicImage:
        return self._parse_image(self._get(f"images/{uuid}", {"build": self.build}))

    def download_image(self, image: PhyloPicImage) -> bytes:
        response = requests.get(image.download_url, timeout=self.timeout)
        response.raise_for_status()
        return response.content

    @property
    def build(self) -> int:
        if self._build is None:
            self._build = self._get("images").get("build")
        return self._build

    def _find_best_node_uuid(self, query: str) -> str | None:
        data = self._get(
            "nodes",
            {
                "build": self.build,
                "page": 0,
                "embed_items": "true",
                "filter_name": query.lower().replace("_", " "),
            },
        )
        items = data.get("_embedded", {}).get("items", [])
        if not items:
            return None

        href = items[0].get("_links", {}).get("self", {}).get("href", "")
        return self.extract_uuid(href)

    def _parse_image(self, data: dict) -> PhyloPicImage:
        links = data.get("_links", {})
        self_link = links.get("self", {})
        uuid = self.extract_uuid(self_link.get("href", "")) or data.get("uuid", "")
        title = self_link.get("title") or self._first_link_title(links.get("nodes")) or uuid
        raster_url = self._choose_raster_url(links.get("rasterFiles", []))
        og_image = links.get("http://ogp.me/ns#image", {}).get("href", "")

        if not uuid or not raster_url:
            raise ValueError("PhyloPic image is missing a UUID or raster download.")

        return PhyloPicImage(
            uuid=uuid,
            title=title,
            page_url=f"https://www.phylopic.org/images/{uuid}/{self._slugify(title)}",
            preview_url=og_image or raster_url,
            download_url=raster_url,
            license_url=links.get("license", {}).get("href", ""),
            contributor=links.get("contributor", {}).get("title", ""),
        )

    def _choose_raster_url(self, rasters: Iterable[dict]) -> str:
        raster_list = list(rasters or [])
        if not raster_list:
            return ""

        def height_delta(raster: dict) -> int:
            sizes = raster.get("sizes", "")
            match = re.search(r"x(\d+)", sizes)
            height = int(match.group(1)) if match else 512
            return abs(height - 512)

        return min(raster_list, key=height_delta).get("href", "")

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = requests.get(
            urljoin(f"{self.base_url}/", path.lstrip("/")),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def extract_uuid(text: str) -> str | None:
        match = UUID_RE.search(text or "")
        return match.group(0).lower() if match else None

    @staticmethod
    def _first_link_title(links) -> str:
        if isinstance(links, list) and links:
            return links[0].get("title", "")
        return ""

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        return slug or "silhouette"
