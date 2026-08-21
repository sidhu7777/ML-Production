"""
Local tile cache + verified/retrying screenshot capture (test-case only;
tools/report_engine is NOT modified).

Two problems traced back to the SAME root cause: every map in this report
fetches its CartoDB Voyager basemap tiles live over the internet during
rendering (map_generator.REPORT_TILE; URL confirmed via folium itself --
https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png),
and production's own tile-load wait in playwright_utils.html_to_png()
explicitly continues on a timeout instead of rejecting or retrying:
    # Last attempt failed, but continue anyway
    print(f"Warning: Tile loading check failed after 3 attempts: {e}")
That combination (live external dependency + no rejection) is what
produced the partially-grey / continent-zoomed screenshots seen in
project 348's report (the continent-zoom part was the separate polygon
axis-order bug, already fixed in _load_report_data; the grey/incomplete
part is this).

Google's Map Tiles API (google_tiles.py) was the originally planned
alternative, but it's blocked on API-key/billing and unavailable per
direction received, so this solves it without any paid API: a small
local caching HTTP proxy in front of the SAME free CartoDB tiles this
report already uses. The first request for a given tile fetches it from
CartoDB once and saves it to disk under output/tile_cache/; every later
request -- same run, or any future run -- is served straight from local
disk, no network round-trip, so rendering can't be slowed or rate-limited
by the tile CDN.

Four pieces:
  1. TileCacheServer (module-level singleton) -- a ThreadingHTTPServer on
     127.0.0.1 serving/caching tiles under output/tile_cache/.
  2. use_local_tiles() -- monkeypatches map_generator.new_report_map /
     grid_rsrp_map_test.new_report_map (identical mechanism to
     google_tiles.use_google_tiles(), including the same "grid_rsrp_map_test
     imports new_report_map at module level so it needs patching directly
     too" exception) to point every map's TileLayer at the local proxy
     instead of CartoDB directly, restoring the originals afterward. It
     ALSO patches map_generator.add_legend -> add_legend_robust (see #4).
  3. html_to_png_verified() -- wraps production's own html_to_png()
     unchanged, but actually inspects the resulting screenshot: if too much
     of it is flat Leaflet-container grey (#ddd, what an unloaded tile pane
     shows through), the render is treated as incomplete and retried from a
     FRESH page load (not just a longer wait) up to MAX_RENDER_RETRIES
     times. Still blank after that -> raises, instead of silently saving a
     broken image into the report.
  4. add_legend_robust() -- a second, independent bug from the same "no
     verification, no retry" pattern: production's add_legend()
     (map_generator.py:184) injects the legend via a ONE-SHOT
     `setTimeout(injectLegend, 250)` with no retry and no confirmation it
     ran. Under load (many maps rendering back-to-back in a full report
     run), that single callback can lose the race against Playwright's
     screenshot -- reproduced directly for project 348's
     handover_map_with_session_legend.py output (map rendered correctly,
     legend box empty). html_to_png_verified()'s blank-detector doesn't
     catch this case because it looks at the WHOLE image and the legend is
     a small corner of an otherwise fully-rendered map. Fix: identical
     legend HTML/CSS/content to production, but injected via an immediate,
     retrying poll (same proven pattern as google_tiles.py's
     initGrayClip(retries)) instead of a single fixed-delay timer -- since
     the legend script tag runs after the map's own init script in DOM
     order, `.folium-map` is normally already present, so this now injects
     synchronously on first attempt instead of racing a 250ms clock.
"""
import contextlib
import http.server
import os
import socket
import threading
from pathlib import Path

from PIL import Image, ImageStat

_TILE_UPSTREAM = "https://a.basemaps.cartocdn.com"
_CACHE_DIR = Path(__file__).parent / "output" / "tile_cache"

_server_lock = threading.Lock()
_server_state: dict = {}  # {"port": int, "httpd": ThreadingHTTPServer}


class _TileCacheHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence per-tile access logging (would be one line per tile)

    def do_GET(self):
        # rel includes the CartoDB style path, e.g.
        # "rastertiles/voyager/12/1234/2345.png" or "light_all/12/.../2345@2x.png"
        # -- kept generic (not hardcoded to one style) so BOTH the base
        # Voyager tiles and the grid maps' gray Positron overlay
        # (google_tiles.new_report_map_gray_clipped) share this same cache.
        rel = self.path.lstrip("/")
        local_path = _CACHE_DIR / rel

        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import urllib.request
                with urllib.request.urlopen(f"{_TILE_UPSTREAM}/{rel}", timeout=15) as resp:
                    data = resp.read()
                # Write via a unique temp file + atomic rename so two
                # concurrent misses for the same tile (a single map page can
                # fire many parallel tile requests) can never interleave
                # into a corrupt cached file.
                tmp_path = local_path.with_name(f"{local_path.name}.tmp{os.getpid()}{threading.get_ident()}")
                tmp_path.write_bytes(data)
                tmp_path.replace(local_path)
            except Exception as exc:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(f"tile fetch failed: {exc}".encode())
                return

        data = local_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _ensure_server() -> int:
    """Lazily starts ONE local caching proxy for the whole process (same
    singleton pattern as google_tiles._SESSION_CACHE), reused by every map
    render in this run. Returns its port."""
    with _server_lock:
        if "port" in _server_state:
            return _server_state["port"]
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _TileCacheHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _server_state["port"] = port
        _server_state["httpd"] = httpd
        return port


def local_tile_url(style: str = "rastertiles/voyager") -> str:
    """
    URL template for a given CartoDB style ("rastertiles/voyager" for the
    normal basemap, "light_all" for the gray Positron style
    google_tiles.new_report_map_gray_clipped overlays on grid maps),
    served through the local caching proxy instead of live CartoDB.
    Starts the proxy on first use.
    """
    port = _ensure_server()
    return f"http://127.0.0.1:{port}/{style}/{{z}}/{{x}}/{{y}}{{r}}.png"


def new_report_map_local():
    """Drop-in replacement for production's new_report_map()
    (tools/report_engine/map_generator.py, NOT modified) -- identical
    folium.Map settings, but its TileLayer points at the local caching
    proxy instead of CartoDB directly."""
    import folium
    from tools.report_engine.map_generator import REPORT_MAP_MAX_ZOOM

    m = folium.Map(
        tiles=None,
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=REPORT_MAP_MAX_ZOOM,
        zoomSnap=0,
        zoomDelta=0.25,
    )
    folium.TileLayer(
        tiles=local_tile_url("rastertiles/voyager"),
        attr="© CartoDB, © OpenStreetMap contributors",
        name="CartoDB Voyager (local cache)",
        overlay=False,
        control=False,
        max_zoom=REPORT_MAP_MAX_ZOOM,
    ).add_to(m)
    return m


def add_legend_robust(m, title, items):
    """
    Drop-in replacement for production's add_legend()
    (tools/report_engine/map_generator.py:102-188, NOT modified) -- BYTE
    -IDENTICAL legend CSS/HTML/content (same classes, same row template,
    same "label : count" text) so the rendered legend is visually
    indistinguishable from production's. The only change is how it gets
    attached to the page: production injects it via a single fixed
    `setTimeout(injectLegend, 250)` with no retry and no confirmation it
    ran; this uses an immediate, retrying poll instead (same proven
    pattern as google_tiles.py's initGrayClip(retries)) -- since this
    script tag runs after the map's own init script in DOM order,
    `.folium-map` is normally already present, so injection now happens
    synchronously on the first attempt instead of racing a 250ms clock
    against Playwright's screenshot. Confirmed root cause + fix for
    project 348's handover map: map rendered correctly, legend box came
    back empty under load; re-rendering the identical saved HTML in
    isolation (no load) always produced the correct legend -- a pure
    timing race, not a content/data bug.
    """
    import json
    import folium

    legend_css = """
    <style>
        .kpi-legend {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 320px;
            max-height: calc(100% - 40px);
            overflow: auto;
            z-index: 9999;
            background-color: rgba(255, 255, 255, 0.98);
            color: #000;
            padding: 18px 16px;
            border-radius: 8px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
            border: 2px solid rgba(0,0,0,0.15);
        }
        .kpi-legend-title {
            font-weight: 700;
            font-size: 20px;
            margin-bottom: 12px;
            color: #000;
            border-bottom: 2px solid #333;
            padding-bottom: 8px;
        }
        .kpi-legend-row {
            margin-top: 10px;
            display: flex;
            align-items: center;
            font-size: 18px;
        }
        .kpi-legend-swatch {
            width: 22px;
            height: 22px;
            border-radius: 3px;
            margin-right: 10px;
            flex: 0 0 auto;
            border: 1px solid rgba(0,0,0,0.2);
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(legend_css))

    rows_html = ""
    for label, color, count in items:
        rows_html += (
            f"<div class='kpi-legend-row'>"
            f"<span class='kpi-legend-swatch' style='background:{color};'></span>"
            f"<span>{label} : {count}</span>"
            f"</div>"
        )

    legend_inner_html = (
        f"<div class='kpi-legend'>"
        f"<div class='kpi-legend-title'>{title}</div>"
        f"{rows_html}"
        f"</div>"
    )

    payload = json.dumps(legend_inner_html)
    legend_js = f"""
    <script>
        (function() {{
            function injectLegend(retries) {{
                var mapEl = document.querySelector('.folium-map');
                if (!mapEl) {{
                    if (retries > 0) {{ setTimeout(function() {{ injectLegend(retries - 1); }}, 100); }}
                    return;
                }}
                mapEl.style.position = 'relative';
                var existing = mapEl.querySelector('.kpi-legend');
                if (existing) existing.remove();
                var wrapper = document.createElement('div');
                wrapper.innerHTML = {payload};
                mapEl.appendChild(wrapper.firstElementChild);
            }}
            injectLegend(50);
        }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(legend_js))


@contextlib.contextmanager
def use_local_tiles():
    """
    Monkeypatches every known reference to production's new_report_map AND
    add_legend so EVERY map in this report picks up the local tile cache
    and the retrying legend injection for the duration of this block --
    same patch targets/mechanism as google_tiles.use_google_tiles():
    map_generator.new_report_map / map_generator.add_legend cover every
    function (production's own + this test case's) that does a fresh
    `from tools.report_engine.map_generator import ...` inside its own
    body, since that resolves the module's current attribute at call
    time. grid_rsrp_map_test.py is the one exception for both -- it
    imports new_report_map AND add_legend at MODULE level, binding its
    own separate names, so both need patching directly too. Restores the
    originals afterward; tools/report_engine is never modified on disk.
    """
    from tools.report_engine import map_generator
    from tests.new_pdf_report import grid_rsrp_map_test

    original_map_mg = map_generator.new_report_map
    original_map_grid = grid_rsrp_map_test.new_report_map
    original_legend_mg = map_generator.add_legend
    original_legend_grid = grid_rsrp_map_test.add_legend
    map_generator.new_report_map = new_report_map_local
    grid_rsrp_map_test.new_report_map = new_report_map_local
    map_generator.add_legend = add_legend_robust
    grid_rsrp_map_test.add_legend = add_legend_robust
    try:
        yield
    finally:
        map_generator.new_report_map = original_map_mg
        grid_rsrp_map_test.new_report_map = original_map_grid
        map_generator.add_legend = original_legend_mg
        grid_rsrp_map_test.add_legend = original_legend_grid


# ---------------------------------------------------------------------
# Verified / retrying screenshot capture
# ---------------------------------------------------------------------

MAX_RENDER_RETRIES = 3
_BLANK_FRACTION_THRESHOLD = 0.6  # >=60% flat Leaflet-grey/white => reject
_LEAFLET_CONTAINER_GREY = (221, 221, 221)  # Leaflet's default .leaflet-container #ddd background
_COLOR_TOLERANCE = 6


_REGIONAL_GREY_THRESHOLD = 0.9  # a single grid cell this dominated by flat #ddd => that patch never loaded
_REGIONAL_GRID_COLS = 6
_REGIONAL_GRID_ROWS = 6


def _is_flat(rgb) -> bool:
    r, g, b = rgb
    if r > 250 and g > 250 and b > 250:
        return True
    gr, gg, gb = _LEAFLET_CONTAINER_GREY
    return abs(r - gr) <= _COLOR_TOLERANCE and abs(g - gg) <= _COLOR_TOLERANCE and abs(b - gb) <= _COLOR_TOLERANCE


def _is_grey(rgb) -> bool:
    r, g, b = rgb
    gr, gg, gb = _LEAFLET_CONTAINER_GREY
    return abs(r - gr) <= _COLOR_TOLERANCE and abs(g - gg) <= _COLOR_TOLERANCE and abs(b - gb) <= _COLOR_TOLERANCE


def _is_mostly_blank(png_path: str) -> bool:
    """True if the screenshot is dominated by Leaflet's unloaded-tile
    background (flat grey #ddd) or plain white -- i.e. tiles never
    actually finished loading before the screenshot was taken, even though
    production's html_to_png() didn't raise for it (see module docstring:
    it deliberately continues on a tile-load timeout)."""
    with Image.open(png_path) as im:
        im = im.convert("RGB").resize((120, 90))  # cheap; exact resolution doesn't matter for this check
        pixels = list(im.getdata())

    blank = sum(1 for p in pixels if _is_flat(p))
    return (blank / len(pixels)) >= _BLANK_FRACTION_THRESHOLD


def _has_blank_tile_region(png_path: str) -> bool:
    """
    True if any localized patch of the screenshot is dominated by
    Leaflet's unloaded-tile grey (#ddd) SPECIFICALLY -- not white, so the
    legend's own intentionally near-white background
    (rgba(255,255,255,0.98), map_generator.add_legend's .kpi-legend CSS)
    is never mistaken for a defect. Catches partial/corner tile-load
    failures _is_mostly_blank()'s whole-image average can't see because
    they're too small a fraction of the total frame -- e.g. the tile
    directly behind the legend (top-right), which loads last since
    Leaflet prioritizes center tiles first. Reproduced directly on project
    348's tech_handover_map.png: fully rendered map + legend, one grey
    rectangle in that exact corner.
    """
    with Image.open(png_path) as im:
        im = im.convert("RGB").resize((120, 90))
        pixels = im.load()

    cell_w = 120 // _REGIONAL_GRID_COLS
    cell_h = 90 // _REGIONAL_GRID_ROWS

    for cy in range(_REGIONAL_GRID_ROWS):
        for cx in range(_REGIONAL_GRID_COLS):
            total = 0
            grey = 0
            for y in range(cy * cell_h, (cy + 1) * cell_h):
                for x in range(cx * cell_w, (cx + 1) * cell_w):
                    total += 1
                    if _is_grey(pixels[x, y]):
                        grey += 1
            if total and (grey / total) >= _REGIONAL_GREY_THRESHOLD:
                return True
    return False


# Every legend in this report sits at the same CSS position
# (map_generator.add_legend's .kpi-legend: top:20px; right:20px;
# width:320px), so this fixed top-right box covers it regardless of which
# map is being checked. Expressed as fractions of the image so it works at
# any render resolution.
_LEGEND_REGION_BOX = (0.68, 0.0, 1.0, 0.42)  # (left, top, right, bottom)
_LEGEND_MIN_STDDEV = 10.0


def _legend_region_looks_empty(png_path: str) -> bool:
    """
    True if the top-right corner is suspiciously flat/uniform -- i.e. a
    legend box rendered its own (legitimately white) background
    successfully, but its title/rows never got appended before the
    screenshot. Different failure mode from _has_blank_tile_region():
    that catches grey UNLOADED tiles; this catches a legend that's just
    empty inside. Reproduced directly on project 348's handover_map.png:
    add_legend_robust's injectLegend ran and the saved HTML had fully
    correct legend content, but under the full multi-map report run's
    load the browser hadn't executed the injection yet by screenshot
    time; re-rendering the identical HTML in isolation (no competing
    load) always produced the correct legend -- a pure timing race, same
    root cause as the tile one, different symptom.

    This is necessarily a heuristic (real legend text/swatches have much
    higher local contrast than a flat box; the small risk is a
    legend-less map whose top-right corner happens to be genuinely flat,
    e.g. open water) -- so callers must treat it as a soft signal, not a
    hard failure: retry a few times, but proceed with a warning rather
    than raising if it's still flat afterward, so a map that never had a
    legend to begin with can never be broken by this check.
    """
    with Image.open(png_path) as im:
        im = im.convert("L")
        w, h = im.size
        left, top, right, bottom = _LEGEND_REGION_BOX
        box = (int(w * left), int(h * top), int(w * right), int(h * bottom))
        region = im.crop(box)
        stddev = ImageStat.Stat(region).stddev[0]
    return stddev < _LEGEND_MIN_STDDEV


def html_to_png_verified(html_path, png_path, **kwargs):
    """
    Drop-in replacement for tools.report_engine.playwright_utils.html_to_png
    (that file is NOT modified) that actually verifies the result instead
    of trusting it: if the screenshot comes back mostly flat grey/white
    (tiles never loaded at all) OR has any localized unloaded-tile grey
    patch (a corner that lost the load race while the rest of the map
    rendered fine -- the exact failure production's own version silently
    accepts either way), the ENTIRE render is retried from a fresh page
    load -- not just waited on longer -- up to MAX_RENDER_RETRIES times.
    Raises if it's still broken after that, so a broken map can never
    silently end up in the PDF.

    A legend that rendered its box but not its content
    (_legend_region_looks_empty) is retried the same way, but treated as
    soft: if it's still flat after every retry, the last render is kept
    with a warning instead of raising, since that same flatness can
    legitimately happen on a map that never had a legend at all.
    """
    from tools.report_engine.playwright_utils import html_to_png

    last_exc = None
    legend_still_empty = False
    for attempt in range(1, MAX_RENDER_RETRIES + 1):
        try:
            html_to_png(html_path, png_path, **kwargs)
        except Exception as exc:
            last_exc = exc
            print(f"[tiles] render attempt {attempt}/{MAX_RENDER_RETRIES} raised: {exc}")
            continue

        if _is_mostly_blank(png_path):
            print(
                f"[tiles] render attempt {attempt}/{MAX_RENDER_RETRIES} produced a mostly-blank "
                f"map ({png_path}) -- tiles didn't finish loading; retrying with a fresh page load."
            )
            continue
        if _has_blank_tile_region(png_path):
            print(
                f"[tiles] render attempt {attempt}/{MAX_RENDER_RETRIES} has an unloaded-tile patch "
                f"({png_path}) -- a corner tile lost the load race; retrying with a fresh page load."
            )
            continue
        if _legend_region_looks_empty(png_path):
            legend_still_empty = True
            print(
                f"[tiles] render attempt {attempt}/{MAX_RENDER_RETRIES} has an empty-looking legend "
                f"corner ({png_path}) -- legend content lost the injection race; retrying with a "
                f"fresh page load."
            )
            continue
        return

    if legend_still_empty:
        print(
            f"[tiles] {png_path}: legend corner still looks empty after {MAX_RENDER_RETRIES} "
            f"attempts -- keeping the last render (this map may simply have no legend)."
        )
        return

    raise RuntimeError(
        f"html_to_png_verified: {png_path} still broken (blank or partially unloaded) after "
        f"{MAX_RENDER_RETRIES} full render attempts -- rejecting rather than shipping a broken map."
        + (f" Last error: {last_exc}" if last_exc else "")
    )
