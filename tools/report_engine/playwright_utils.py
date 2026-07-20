# src/playwright_utils.py

import os
import tempfile

from playwright.sync_api import sync_playwright


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


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
    render_timeout_ms = _env_int("REPORT_RENDER_TIMEOUT_MS", 120000)
    navigation_attempts = max(1, _env_int("REPORT_RENDER_NAV_ATTEMPTS", 2))

    with sync_playwright() as p:
        browser = _launch_chromium(p)
        context = None
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=device_scale_factor,
            )
            context.set_default_timeout(render_timeout_ms)
            context.set_default_navigation_timeout(render_timeout_ms)
            page = context.new_page()
            page.set_default_timeout(render_timeout_ms)
            page.set_default_navigation_timeout(render_timeout_ms)

            # Large Folium files can take more than Playwright's default 30s to
            # parse. Retry once before failing the report.
            last_nav_error = None
            for attempt in range(navigation_attempts):
                try:
                    page.goto(
                        html_url,
                        wait_until="domcontentloaded",
                        timeout=render_timeout_ms,
                    )
                    last_nav_error = None
                    break
                except Exception as exc:
                    last_nav_error = exc
                    if attempt + 1 < navigation_attempts:
                        print(
                            f"Warning: map navigation timed out, retrying "
                            f"({attempt + 1}/{navigation_attempts}) for {html_path}: {exc}"
                        )
                        page.close()
                        page = context.new_page()
                        page.set_default_timeout(render_timeout_ms)
                        page.set_default_navigation_timeout(render_timeout_ms)
            if last_nav_error is not None:
                raise last_nav_error
        
            # Wait for the map container to be present
            page.wait_for_selector(".folium-map, .leaflet-container", timeout=render_timeout_ms)

            # Wait for Folium/Leaflet when possible, but do not fail the report if
            # offline tiles/scripts prevent Leaflet from setting its internal flag.
            try:
                page.wait_for_function(
                    """() => {
                        const el = document.querySelector('.folium-map');
                        const map = el && el.id && window[el.id];
                        return !!(map && map._loaded);
                    }""",
                    timeout=min(render_timeout_ms, 30000),
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
                        timeout=min(render_timeout_ms, 20000),
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
                page.wait_for_load_state("networkidle", timeout=min(render_timeout_ms, 10000))
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
                        return

            page.screenshot(path=png_path, full_page=True)
        finally:
            if context is not None:
                context.close()
            browser.close()


def check_chromium_rendering():
    """Return whether the report renderer can launch Chromium and capture a PNG."""
    html_fd, html_path = tempfile.mkstemp(prefix="report-render-health.", suffix=".html")
    png_path = html_path.replace(".html", ".png")
    try:
        with os.fdopen(html_fd, "w", encoding="utf-8") as f:
            f.write(
                "<!doctype html><html><body>"
                "<div class='leaflet-container' "
                "style='width:320px;height:180px;background:#2563eb;color:white'>ok</div>"
                "</body></html>"
            )
        old_timeout = os.environ.get("REPORT_RENDER_TIMEOUT_MS")
        os.environ["REPORT_RENDER_TIMEOUT_MS"] = "5000"
        html_to_png(html_path, png_path, width=320, height=180, device_scale_factor=1, clip_to_map=False)
        if not os.path.exists(png_path) or os.path.getsize(png_path) <= 0:
            return False, "Chromium launched but screenshot file was not created"
        return True, "Chromium renderer is working"
    except Exception as exc:
        return False, str(exc)
    finally:
        for path in (html_path, png_path):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        if "old_timeout" in locals():
            if old_timeout is None:
                os.environ.pop("REPORT_RENDER_TIMEOUT_MS", None)
            else:
                os.environ["REPORT_RENDER_TIMEOUT_MS"] = old_timeout
