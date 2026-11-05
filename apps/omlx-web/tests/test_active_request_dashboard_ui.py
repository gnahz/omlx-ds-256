# SPDX-License-Identifier: Apache-2.0
"""Exercise the Active Models rows with the real Alpine template."""

import json
import os
from pathlib import Path

import pytest
from jinja2 import Environment


WEB = Path(__file__).resolve().parents[1] / "omlx_web"


@pytest.mark.parametrize("width,language", [(1440, "en"), (900, "en"), (390, "en"), (1440, "zh"), (900, "zh"), (390, "zh")])
def test_active_request_handoff_and_recent_decode_rate(width, language):
    playwright = pytest.importorskip("playwright.sync_api")
    locale = json.loads((WEB / f"i18n/{language}.json").read_text())
    if language == "en" and width == 900:
        # Existing running processes keep their locale dictionary until restart.
        locale.pop("status.active_models.total_decode_rate")
    template = Environment(autoescape=True).from_string(
        (WEB / "templates/dashboard/blocks/_active_models.html").read_text()
    )
    markup = template.render(t=lambda key: locale.get(key, key), current_lang=language)
    def label(key):
        return locale["status.active_models." + key]
    def recent(value):
        return label("recent_decode_rate").replace("{rate}", value)
    request_id = "12345678-request"
    model = {
        "id": "deepseek-v41-lsq",
        "request_limit": 4,
        "admitted_requests": 4,
        "queued_requests": 11,
        "active_requests": 1,
        "waiting_requests": 0,
        "prefilling": [],
        "waiting": [],
        "generating": [],
        "activities": [
            {
                "kind": "request_preparation",
                "phase": "preparing_request",
                "request_id": request_id,
            }
        ],
    }
    gib = 1024**3
    model["memory_budget"] = {
        "system_free_bytes": 10 * 1024**3, "system_free_reserve_bytes": 8 * 1024**3,
        "current_bytes": 232*gib, "reserved_bytes": 2*gib, "shared_transient_bytes": gib,
        "available_bytes": 4*gib, "limit_bytes": 238*gib,
        "process_limit_bytes": 240*gib,
        "pending": {"request_id": "oldest00-pending", "shortfall_bytes": 3*gib, "required_bytes": 8*gib,
                    "envelope": {"total_bytes": 7*gib, "context_tokens": 524724, "prefill_chunk_tokens": 256}},
        "reservations": [{"request_id": "decode00-running", "phase": "decode",
                          "reserved_bytes": 2*gib, "prefill_chunk_tokens": 2048}],
    }
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": width, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(
            "http://dashboard.test/",
            lambda route: route.fulfill(
                body=f'<div id="panel" x-data="dashboardFixture()">{markup}</div>',
                content_type="text/html; charset=utf-8",
            ),
        )
        page.goto("http://dashboard.test/")
        page.add_style_tag(path=WEB / "static/css/theme.css")
        page.add_style_tag(path=WEB / "static/css/tailwind.css")
        page.add_script_tag(path=WEB / "static/js/dashboard.js")
        page.evaluate(
            """([locale, model]) => {
                window.t = (key) => locale[key] ?? key;
                window.dashboardFixture = () => {
                    const dashboardData = dashboard();
                    dashboardData.init = () => {};
                    dashboardData.stats.active_models = {
                        models: [model],
                        total_active_requests: 1,
                        total_waiting_requests: 0,
                    };
                    return dashboardData;
                };
            }""",
            [locale, model],
        )
        page.add_script_tag(path=WEB / "static/js/alpine.min.js")
        page.wait_for_function("Alpine.$data(document.querySelector('#panel')).stats")
        assert not errors

        playwright.expect(page.get_by_text(label("pipeline_slots") + ": 4 / 4", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text(label("entrance_queue") + ": 11", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text(label("preparing"), exact=True)).to_be_visible()
        page.evaluate(
            """(requestId) => {
                const row = Alpine.$data(document.querySelector('#panel'))
                    .stats.active_models.models[0];
                row.activities = [];
                row.generating = [{
                    request_id: requestId,
                    generated_tokens: 30,
                    tokens_per_second: 100,
                    recent_tokens_per_second: null,
                }];
            }""",
            request_id,
        )
        playwright.expect(page.get_by_text(label("preparing"), exact=True)).to_have_count(0)
        playwright.expect(page.get_by_text("12345678", exact=True)).to_have_count(1)
        playwright.expect(page.get_by_text(recent("—"), exact=True).last).to_be_visible()
        for rate in (7.5, 0.0):
            page.evaluate(
                """(rate) => {
                    Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0]
                        .generating[0].recent_tokens_per_second = rate;
                }""",
                rate,
            )
            playwright.expect(
                page.get_by_text(recent(f"{rate:.1f} tok/s"), exact=True).last
            ).to_be_visible()
        for field in ("current", "reserved", "available", "limit", "process_limit"):
            field_label = locale["status.active_models.budget_" + field]
            value = model["memory_budget"][field + "_bytes"] / gib
            if field in ("limit", "process_limit"):
                playwright.expect(page.get_by_text(f"{field_label}: {value:.2f} GiB", exact=True)).to_be_visible()
            else:
                playwright.expect(page.locator("[data-budget-usage]").get_by_text(field_label, exact=True)).to_be_visible()
                playwright.expect(page.locator("[data-budget-usage]").get_by_text(f"{value:.2f} GiB", exact=True)).to_be_visible()
        playwright.expect(page.get_by_text(locale["status.active_models.budget_shared"] + ": 1.00 GiB", exact=True)).to_be_visible()
        playwright.expect(page.locator("[data-budget-pending]").get_by_text("8.00 GiB", exact=True)).to_be_visible()
        shortfall = locale["status.active_models.budget_shortfall"]
        playwright.expect(page.locator("[data-budget-pending]").get_by_text(shortfall, exact=True)).to_be_visible()
        total = page.locator("[data-total-decode-rate]")
        for rates, expected in (([14.7, 26, 15.3, 26], "82.0 tok/s"),
                                ([17, 39.7, 15.7], "72.4 tok/s"),
                                ([17, None], "—"), ([0, 0], "0.0 tok/s"),
                                ([], "0.0 tok/s"), ([17, 39.7, 15.7], "72.4 tok/s")):
            page.evaluate("""rates => {
                const m = Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0];
                m.generating = rates.map((rate, i) => ({
                    request_id: 'request' + i + '-decode', recent_tokens_per_second: rate,
                    tokens_per_second: 999, generated_tokens: 3300 + i, elapsed_seconds: 60,
                }));
            }""", rates)
            playwright.expect(total).to_have_text(recent(expected))
        playwright.expect(page.locator("[data-budget-prefill-chunk]")).to_have_text(label("budget_prefill_chunk") + ": 2048 tok")
        # Check actual computed spacing against the shipped CSS, not just classes.
        playwright.expect(page.locator("[data-budget-system]")).to_contain_text("10.00 GiB")
        playwright.expect(page.locator("[data-budget-system]")).to_contain_text("8.00 GiB")
        for selector in ("[data-budget-system]", "[data-budget-usage]", "[data-budget-pending-details]", "[data-budget-limits]"):
            assert page.locator(selector).evaluate("el => parseFloat(getComputedStyle(el).columnGap)") >= 20
        boxes = page.locator("[data-budget-usage] > div").evaluate_all(
            "els => els.map(el => { const b=el.getBoundingClientRect(); return {x:b.x,y:b.y,right:b.right,bottom:b.bottom}; })")
        for left, right in zip(boxes, boxes[1:]):
            assert right["x"] >= left["right"] + 19 or right["y"] >= left["bottom"] + 9
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.evaluate("""() => {
            const data=Alpine.$data(document.querySelector('#panel')).stats.active_models;
            const m=data.models[0];
            m.queued_requests=2; m.active_requests=3; m.waiting_requests=1;
            m.waiting=[{request_id:'oldest00-pending',queue_position:1,elapsed_seconds:3,prompt_tokens:55000,reason:'context_budget'}];
            data.total_active_requests=3; data.total_waiting_requests=1; data.total_queued_requests=2;
        }""")
        if os.environ.get("OMLX_UI_SCREENSHOT_DIR"):
            directory=Path(os.environ["OMLX_UI_SCREENSHOT_DIR"]);directory.mkdir(parents=True,exist_ok=True)
            page.locator("#panel").screenshot(path=directory / f"dashboard-{language}-{width}.png")
        page.evaluate("""() => {
            const budget = Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0].memory_budget;
            budget.pending = null;
            budget.reserved_bytes = 0;
            budget.reservations = [];
        }""")
        playwright.expect(page.locator("[data-budget-pending]").get_by_text(shortfall, exact=True)).to_have_count(0)
        playwright.expect(page.get_by_text("decode00", exact=False)).to_have_count(0)
        page.evaluate("""() => {
            const budget = Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0].memory_budget;
            budget.decode_guard = {state:'waiting_release',last_failure:{event_id:7,shortfall_bytes:1024,victim_request_id:'newest-full-request-id'}};
        }""")
        guard=page.locator('[data-decode-budget-guard]')
        playwright.expect(guard.get_by_text(label('decode_guard_waiting_release'),exact=True)).to_be_visible()
        guard.locator('summary').click()
        playwright.expect(guard.locator('pre')).to_contain_text('newest-full-request-id')
        for state in ('releasing_latest','running'):
            page.evaluate("""state => {
                Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0].memory_budget.decode_guard.state=state;
            }""",state)
            if state!='running':
                playwright.expect(guard.get_by_text(label('decode_guard_'+state),exact=True)).to_be_visible()
        playwright.expect(guard.get_by_text(label('decode_guard_running'),exact=True)).to_be_hidden()
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.evaluate("""() => {
            const b=Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0].memory_budget;
            b.prefill_guard={waiting_request_ids:['preserved'],last_failure:{request_id:'preserved',kv_tokens:62432,stage:'waiting_release'}};
        }""")
        guard=page.locator('[data-prefill-budget-guard]')
        playwright.expect(guard.get_by_text(label('prefill_guard_waiting'),exact=True)).to_be_visible()
        guard.locator('summary').click()
        playwright.expect(guard.locator('pre')).to_contain_text('62432')
        page.evaluate("""() => {Alpine.$data(document.querySelector('#panel')).stats.active_models.models[0].memory_budget.prefill_guard.waiting_request_ids=[];}""")
        playwright.expect(guard.get_by_text(label('prefill_guard_waiting'),exact=True)).to_be_hidden()
        browser.close()
    assert not errors
