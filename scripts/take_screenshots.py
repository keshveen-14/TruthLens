"""Regenerate the README screenshots (docs/screenshots/*.png) from the running UI.

    pip install playwright && playwright install chromium
    python scripts/take_screenshots.py

Starts its own server on a free port. The browser engine is used, so the images are reproducible.
"""

from __future__ import annotations

import os
import sys
import threading

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import server  # noqa: E402

OUT = os.path.join(ROOT, "docs", "screenshots")


def main():
    from playwright.sync_api import sync_playwright

    os.makedirs(OUT, exist_ok=True)
    srv = server.create_server("127.0.0.1", 0, os.path.join(ROOT, "data", "corpus"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    settings = "localStorage.setItem('truthlens.settings', JSON.stringify({engine:'browser',threshold:0.7,k:5}))"

    with sync_playwright() as p:
        browser = p.chromium.launch()

        def open_page(hash_="#ask", **kw):
            page = browser.new_context(**kw).new_page()
            page.add_init_script(settings)
            page.goto(url + hash_)
            page.wait_for_timeout(3200)  # fonts, then the verdict animation
            return page

        desktop = dict(viewport={"width": 1360, "height": 900}, device_scale_factor=2)
        page = open_page(**desktop, color_scheme="light")
        page.evaluate("window.scrollTo(0, document.querySelector('.verdict-title').getBoundingClientRect().top + scrollY - 90)")
        page.screenshot(path=os.path.join(OUT, "answer-light.png"))
        page.click("[role=tab]:has-text('After fixing')")
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(OUT, "after-fixing.png"))
        page = open_page(**desktop, color_scheme="dark")
        page.evaluate("window.scrollTo(0, document.querySelector('.verdict-title').getBoundingClientRect().top + scrollY - 90)")
        page.screenshot(path=os.path.join(OUT, "answer-dark.png"))
        page = open_page("#benchmark", **desktop, color_scheme="light")
        page.click("#bench-run")
        page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(OUT, "stress-test.png"), full_page=True)
        open_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                  color_scheme="light").screenshot(path=os.path.join(OUT, "mobile.png"), full_page=True)
        browser.close()
    srv.shutdown()
    print(f"wrote screenshots to {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
