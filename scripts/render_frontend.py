"""Render the Stage 10 frontend and screenshot it, to confirm the UI actually works.

The API and the Vite proxy were already verified with curl, but curl cannot tell
you whether React mounted, whether the tabs render, or whether the form validates.
This drives a real browser through all three tabs and writes screenshots, so the
frontend is checked rather than assumed.

Run:  python scripts/render_frontend.py [--url http://127.0.0.1:5173] [--out reports/screenshots]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

TABS = ["Score a transaction", "Review queue", "About the model"]

NIGHT_FRAUD = {
    "Transaction date & time": "2020-06-21T23:44",
    "Amount (USD)": "1024.55",
    "Card number": "371449635398431",
    "Cardholder latitude": "36.0788",
    "Cardholder longitude": "-81.1781",
    "City population": "3495",
    "Date of birth": "1988-03-09",
    "Merchant latitude": "36.0113",
    "Merchant longitude": "-82.0483",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173")
    parser.add_argument("--out", default=str(ROOT / "reports" / "screenshots"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})

        console_errors: list[str] = []
        page.on("console", lambda message: console_errors.append(message.text)
                if message.type == "error" else None)
        page.on("pageerror", lambda error: console_errors.append(str(error)))

        page.goto(args.url, wait_until="networkidle")
        page.wait_for_selector("h1", timeout=15000)
        print(f"loaded {args.url}: {page.title()!r}")

        status = page.locator(".status").inner_text()
        print(f"status badge: {status!r}")
        if "online" not in status.lower():
            failures.append(f"service badge says {status!r}, expected online")

        page.screenshot(path=str(out / "01-score-tab.png"), full_page=True)

        # 1. score a transaction and read the verdict off the page
        for label, value in NIGHT_FRAUD.items():
            field = page.locator(f'#field-{label_to_name(label)}')
            if field.count():
                field.fill(value)
        selects = page.locator("select")
        if selects.count() >= 4:
            for index, option_value in enumerate(["fraud_Rippin, Kub and Mann",
                                                 "gas_transport", "F", "NC"]):
                selects.nth(index).select_option(option_value)

        page.get_by_role("button", name="Score transaction").click()
        try:
            page.wait_for_selector(".verdict", timeout=20000)
        except Exception as error:  # noqa: BLE001
            failures.append(f"no verdict rendered: {error}")
        else:
            verdict = page.locator(".verdict-label").inner_text()
            score = page.locator(".verdict .score").inner_text()
            print(f"verdict: {verdict!r} at {score}")
            if "fraud" not in verdict.lower():
                failures.append(f"night/gas/$1024 should flag as fraud, got {verdict!r}")
            if page.locator(".reasons li").count() == 0:
                failures.append("verdict rendered with no reasons")
        page.screenshot(path=str(out / "02-verdict-fraud.png"), full_page=True)

        # 2. client-side validation must block a blank form before any request.
        # The form arrives pre-filled with a worked example, so clear it first.
        page.get_by_role("tab", name="Review queue").click()
        page.wait_for_timeout(400)
        for field in page.locator("input").all():
            field.fill("")
        for select in page.locator("select").all():
            select.select_option("")
        page.get_by_role("button", name="Score and add").click()
        page.wait_for_timeout(600)
        errors = page.locator(".field-error").count()
        print(f"blank-submit validation errors shown: {errors}")
        if errors == 0:
            failures.append("submitting a blank form showed no field errors")
        page.screenshot(path=str(out / "03-validation.png"), full_page=True)

        # 3. the model card tab must show real metrics, not zeros
        page.get_by_role("tab", name="About the model").click()
        page.wait_for_selector(".importances li", timeout=15000)
        facts = page.locator(".facts dd").all_inner_texts()
        print(f"model card facts: {facts[:6]}")
        if not any(value not in ("", "0", "0.0000") for value in facts):
            failures.append("model card shows no metrics")
        if page.locator(".importances li").count() < 5:
            failures.append("feature importance list is empty")
        page.screenshot(path=str(out / "04-model-tab.png"), full_page=True)

        # 4. narrow viewport, to confirm the layout is not desktop-only
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(300)
        page.get_by_role("tab", name="Score a transaction").click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "05-mobile.png"), full_page=True)
        print("captured mobile viewport")

        browser.close()

    if console_errors:
        failures.append(f"console errors: {console_errors[:5]}")

    print()
    for path in sorted(out.glob("*.png")):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size:,} bytes")

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("frontend rendered and all checks passed")
    return 0


def label_to_name(label: str) -> str:
    """'Transaction date & time' -> 'trans_date_trans_time'."""
    mapping = {
        "Transaction date & time": "trans_date_trans_time",
        "Amount (USD)": "amt",
        "Card number": "cc_num",
        "Cardholder latitude": "lat",
        "Cardholder longitude": "long",
        "City population": "city_pop",
        "Date of birth": "dob",
        "Merchant latitude": "merch_lat",
        "Merchant longitude": "merch_long",
    }
    return mapping[label]


if __name__ == "__main__":
    sys.exit(main())
