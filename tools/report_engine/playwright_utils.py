# src/playwright_utils.py

import os

from playwright.sync_api import sync_playwright


def _candidate_browser_paths():
    env_path = os.getenv("REPORT_CHROMIUM_PATH") or os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if env_path:
        yield env_path

    local_app_data = os.getenv("LOCALAPPDATA") or ""
    program_files = os.getenv("PROGRAMFILES") or r"C:\Program Files"
    program_files_x86 = os.getenv("PROGRAMFILES(X86)") or r"C:\Program Files (x86)"

    yield os.path.join(program_files, "Microsoft", "Edge", "Application", "msedge.exe")
    yield os.path.join(program_files_x86, "Microsoft", "Edge", "Application", "msedge.exe")
    yield os.path.join(program_files, "Google", "Chrome", "Application", "chrome.exe")
    yield os.path.join(program_files_x86, "Google", "Chrome", "Application", "chrome.exe")
    if local_app_data:
        yield os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe")


def _launch_chromium(playwright):
    try:
        return playwright.chromium.launch()
    except Exception as first_error:
        launch_errors = [str(first_error)]

    for channel in ("msedge", "chrome"):
        try:
            return playwright.chromium.launch(channel=channel)
        except Exception as exc:
            launch_errors.append(str(exc))

    for browser_path in _candidate_browser_paths():
        if not browser_path or not os.path.exists(browser_path):
            continue
        try:
            return playwright.chromium.launch(executable_path=browser_path)
        except Exception as exc:
            launch_errors.append(f"{browser_path}: {exc}")

    raise RuntimeError(
        "Unable to launch a Chromium browser for report rendering. "
        "Run `ML\\venv\\Scripts\\python.exe -m playwright install chromium` "
        "or set REPORT_CHROMIUM_PATH to chrome.exe/msedge.exe. "
        f"Launch errors: {' | '.join(launch_errors)}"
    )


def html_to_png(
    html_path,
    png_path,
    width=1920,
    height=1200,
    device_scale_factor=2,  # Reduced from 3 to 2 for smaller file size
    clip_to_map=True,
):
    html_path = os.path.abspath(html_path)
    html_url = "file:///" + html_path.replace("\\", "/")

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()

        # Load the page and wait for DOM ready
        page.goto(html_url, wait_until="domcontentloaded")
        
        # Wait for the map container to be present
        page.wait_for_selector(".folium-map, .leaflet-container", timeout=30000)

        # Wait for Folium/Leaflet when possible, but do not fail the report if
        # offline tiles/scripts prevent Leaflet from setting its internal flag.
        try:
            page.wait_for_function(
                """() => {
                    const el = document.querySelector('.folium-map');
                    const map = el && el.id && window[el.id];
                    return !!(map && map._loaded);
                }""",
                timeout=15000,
            )
        except Exception as e:
            print(f"Warning: Leaflet map load check timed out, continuing screenshot: {e}")

        # Invalidate map size to ensure proper rendering
        try:
            page.evaluate(
                """() => {
                    const el = document.querySelector('.folium-map');
                    const map = el && el.id && window[el.id];
                    if (map && typeof map.invalidateSize === 'function') {
                        map.invalidateSize(true);
                    }
                }"""
            )
        except Exception as e:
            print(f"Warning: Map invalidateSize failed, continuing screenshot: {e}")

        # Wait for tiles to load - flexible approach
        # Check multiple times to ensure tiles are actually loaded
        for attempt in range(3):
            try:
                page.wait_for_function(
                    """() => {
                        const loaded = document.querySelectorAll('.leaflet-tile-loaded').length;
                        const loading = document.querySelectorAll('.leaflet-tile-loading').length;
                        // More flexible: require at least some tiles loaded and no loading tiles
                        return loaded >= 10 && loading === 0;
                    }""",
                    timeout=15000,
                )
                # If successful, break the loop
                break
            except Exception as e:
                if attempt < 2:
                    # Wait a bit and retry
                    page.wait_for_timeout(2000)
                else:
                    # Last attempt failed, but continue anyway
                    print(f"Warning: Tile loading check failed after 3 attempts: {e}")

        # Additional wait for network to be idle (all tile requests complete)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception as e:
            # Network idle not critical, continue anyway
            pass

        # Final settling delay to ensure all rendering is complete
        page.wait_for_timeout(2000)

        # Force one more map refresh
        try:
            page.evaluate(
                """() => {
                    const el = document.querySelector('.folium-map');
                    const map = el && el.id && window[el.id];
                    if (map && typeof map.invalidateSize === 'function') {
                        map.invalidateSize(true);
                    }
                }"""
            )
        except Exception as e:
            print(f"Warning: Final map refresh failed, continuing screenshot: {e}")

        # Small delay after final refresh
        page.wait_for_timeout(500)

        if clip_to_map:
            map_el = page.query_selector(".folium-map") or page.query_selector(".leaflet-container")
            if map_el:
                box = map_el.bounding_box()
                if box:
                    page.screenshot(
                        path=png_path,
                        clip={
                            "x": box["x"],
                            "y": box["y"],
                            "width": box["width"],
                            "height": box["height"],
                        },
                    )
                    context.close()
                    browser.close()
                    return

        page.screenshot(path=png_path, full_page=True)
        context.close()
        browser.close()
