import pytest
from playwright.sync_api import sync_playwright

def test_navbar_and_navigation():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:5050")  # Adjust port as needed

        # Check navbar exists
        assert page.locator("nav").is_visible()
        # Check all main nav buttons are present
        for label in ["Models Library", "Hugging Face", "Settings"]:
            assert page.get_by_role("link", name=label).is_visible()

        # Click each nav button and check page changes
        page.get_by_role("link", name="Hugging Face").click()
        assert "huggingface" in page.url.lower()

        page.get_by_role("link", name="Settings").click()
        assert "settings" in page.url.lower()

        browser.close()
