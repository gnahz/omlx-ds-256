"""Render custom exact-cache accounting together with upstream SSD warnings."""

import json
from pathlib import Path

import pytest
from jinja2 import Environment

WEB = Path(__file__).resolve().parents[1] / "omlx_web"


@pytest.mark.parametrize("width", [390, 1440])
def test_exact_cache_checkpoints_and_unavailable_warning_coexist(width):
    playwright = pytest.importorskip("playwright.sync_api")
    locale = json.loads((WEB / "i18n/en.json").read_text())
    markup = Environment(autoescape=True).from_string(
        (WEB / "templates/dashboard/blocks/_cache_observability.html").read_text()
    ).render(t=lambda key: locale.get(key, key))
    cache = {
        "backend": "deepseek_v41_exact_l2_v1", "ssd_cache_unavailable": True,
        "ssd_cache_dir": "/unavailable/exact-cache", "base_path": "/runtime",
        "hot_cache_max_bytes": 2048, "hot_cache_size_bytes": 1024,
        "hot_cache_entries": 7, "total_size_bytes": 4096,
        "disk_max_bytes": 8192, "total_num_files": 9,
        "models": [{
            "id": "deepseek-v41-lsq", "cache_tier": "deepseek-exact-l2",
            "backend": "deepseek_v41_exact_l2_v1", "block_size": 0,
            "indexed_blocks": 0, "exact_prefix_cache": {"checkpoints": 7},
            "num_files": 9, "total_size_bytes": 4096,
            "hot_cache_entries": 7, "hot_cache_size_bytes": 1024,
            "hits": 3, "misses": 4, "loads": 5, "saves": 6,
            "read_bytes": 700, "write_bytes": 800, "pending_bytes": 900,
        }],
    }
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://dashboard.test/", lambda route: route.fulfill(
            body=f'<div id="panel" x-data="cacheFixture()">{markup}</div>',
            content_type="text/html; charset=utf-8"))
        page.goto("http://dashboard.test/")
        page.add_style_tag(path=WEB / "static/css/theme.css")
        page.add_style_tag(path=WEB / "static/css/tailwind.css")
        page.add_script_tag(path=WEB / "static/js/dashboard.js")
        page.evaluate("""([locale, cache]) => {
            window.t = key => locale[key] ?? key;
            window.cacheFixture = () => {
                const app = dashboard();
                app.init = () => {};
                app.stats.runtime_cache = cache;
                return app;
            };
        }""", [locale, cache])
        page.add_script_tag(path=WEB / "static/js/alpine.min.js")
        playwright.expect(page.get_by_text("Exact CED · RAM + SSD L2", exact=True)).to_be_visible()
        warning = locale["status.runtime_cache.ssd_unavailable"].replace("{path}", cache["ssd_cache_dir"])
        playwright.expect(page.get_by_text(warning, exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("7 checkpoints", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text("0 blocks", exact=True)).to_have_count(0)
        playwright.expect(page.get_by_text("SSD hits 3 · misses 4", exact=False)).to_be_visible()
        page.evaluate("""() => {
            const cache = Alpine.$data(document.querySelector('#panel')).stats.runtime_cache;
            cache.ssd_cache_unavailable = false;
            cache.models[0].exact_prefix_cache.checkpoints = 8;
        }""")
        playwright.expect(page.get_by_text(warning, exact=True)).to_be_hidden()
        playwright.expect(page.get_by_text("8 checkpoints", exact=True)).to_be_visible()
        assert not errors
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        browser.close()
