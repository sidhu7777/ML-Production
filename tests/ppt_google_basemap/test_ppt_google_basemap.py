"""
Test case for the Mobility DT PPT report with a Google Maps basemap
instead of CartoDB Voyager. Production code (tools/Ppt_report_Automation,
tools/report_engine) is NOT modified.

Why: CartoDB now serves "API KEY REQUIRED" watermark tiles, which show up
on every PPT map slide.

How: GOOGLE_MAPS_API_KEY (ML/.env) has the Maps Static API enabled but NOT
the Map Tiles API, so this test uses Static Maps images (the same API as
Research/open_map.py) instead of an XYZ tile layer:
  * new_report_map() is swapped at runtime (restored afterwards) for the
    same folium map with no tile layer, plus a small script.
  * After Leaflet has fitted the data bounds (same fractional zoom, same Web
    Mercator projection, same framing as production), the script requests a
    grid of 640x640 @2x Static Maps images at floor(zoom) covering the
    viewport and places each one as an L.imageOverlay at its exact
    geographic bounds. Rows overlap by 40 px so each image's Google logo
    strip is hidden under the row below; the bottom row keeps it visible.
  * The overlays carry Leaflet's tile classes, so production's Playwright
    tile-load wait (report_engine/playwright_utils.html_to_png) works as is.
  * Every rendered map PNG is also copied to output/maps/ for review
    (production deletes its temp render folder when the run finishes).

Run standalone (from the ML folder):
    venv\\Scripts\\python.exe -B tests\\ppt_google_basemap\\test_ppt_google_basemap.py --project-id 363 --country-code india
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

# --- Bootstrap: same import layout and .env order as Ppt_report_Automation/main.py ---
ML_ROOT = Path(__file__).resolve().parents[2]
PPT_DIR = ML_ROOT / "tools" / "Ppt_report_Automation"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MAPS_DIR = OUTPUT_DIR / "maps"

for _p in (str(ML_ROOT), str(PPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PPT_DIR / ".env")  # production PPT settings win (loaded first)
load_dotenv(ML_ROOT / ".env")  # adds GOOGLE_MAPS_API_KEY

DEFAULT_PROJECT_ID = 363
DEFAULT_COUNTRY = "india"

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
STATIC_IMG_PX = 640    # Static Maps max logical image size
LOGO_OVERLAP_PX = 40   # hides the Google logo/copyright strip of the row above
STATIC_MAX_ZOOM = 21

_BASEMAP_JS = """
<script>
(function () {
  var MAP_NAME = %(map_name)s;
  var BASE_URL = %(base_url)s;
  var KEY = %(key)s;
  var IMG = %(img)d, OVERLAP = %(overlap)d, MAX_Z = %(max_z)d;
  var PANE = 'googleStaticBasemap';
  var overlays = [];
  var rebuildTimer = null;

  function markDone(evt) {
    var img = evt.target.getElement();
    if (!img) return;
    L.DomUtil.removeClass(img, 'leaflet-tile-loading');
    L.DomUtil.addClass(img, 'leaflet-tile-loaded');
  }

  function build(map) {
    overlays.forEach(function (o) { map.removeLayer(o); });
    overlays = [];

    var z = map.getZoom();
    var zi = Math.max(0, Math.min(MAX_Z, Math.floor(z)));
    var k = Math.pow(2, zi - z);  // current-zoom pixels -> zoom-zi pixels
    var pb = map.getPixelBounds();
    var minX = pb.min.x * k, maxX = pb.max.x * k;
    var minY = pb.min.y * k, maxY = pb.max.y * k;

    var stepY = IMG - OVERLAP;
    var cols = Math.max(1, Math.ceil((maxX - minX) / IMG));
    var rows = Math.max(1, Math.ceil((maxY - minY - IMG) / stepY) + 1);
    var y0Bottom = maxY - IMG;  // bottom row flush with the viewport bottom

    // Top -> bottom, so each lower row is drawn over the logo strip above it.
    for (var r = 0; r < rows; r++) {
      var y0 = y0Bottom - (rows - 1 - r) * stepY;
      for (var c = 0; c < cols; c++) {
        var x0 = minX + c * IMG;
        var center = map.unproject(L.point(x0 + IMG / 2, y0 + IMG / 2), zi);
        var nw = map.unproject(L.point(x0, y0), zi);
        var se = map.unproject(L.point(x0 + IMG, y0 + IMG), zi);
        var url = BASE_URL
          + '?center=' + center.lat.toFixed(7) + ',' + center.lng.toFixed(7)
          + '&zoom=' + zi + '&size=' + IMG + 'x' + IMG
          + '&scale=2&maptype=roadmap&format=png&language=en'
          + '&key=' + encodeURIComponent(KEY);
        var ov = L.imageOverlay(url, L.latLngBounds(se, nw), {
          pane: PANE,
          interactive: false,
          className: 'leaflet-tile leaflet-tile-loading'
        });
        ov.on('load', markDone);
        ov.on('error', function (evt) {
          markDone(evt);
          console.error('Google Static Maps image failed to load');
        });
        ov.addTo(map);
        overlays.push(ov);
      }
    }
  }

  function start() {
    var map = window[MAP_NAME];
    if (!window.L || !map || !map._loaded) {
      window.setTimeout(start, 50);
      return;
    }
    if (!map.getPane(PANE)) {
      var pane = map.createPane(PANE);
      pane.style.zIndex = 150;
      pane.style.pointerEvents = 'none';
    }
    build(map);
    map.on('moveend zoomend resize', function () {
      window.clearTimeout(rebuildTimer);
      rebuildTimer = window.setTimeout(function () { build(map); }, 100);
    });
  }

  start();
})();
</script>
"""


def google_api_key() -> str:
    key = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set in ML/.env")
    return key


def check_static_maps_key(api_key: str) -> None:
    """Fail fast, before the long DB + render pipeline, if the key can't fetch an image."""
    import requests

    try:
        resp = requests.get(
            STATIC_MAPS_URL,
            params={"center": "28.6139,77.2090", "zoom": 12, "size": "64x64", "key": api_key},
            timeout=20,
        )
    except requests.RequestException as exc:
        # requests puts the full URL (with the key) in its message
        raise RuntimeError(
            f"Google Static Maps check failed: {type(exc).__name__}: {str(exc).replace(api_key, '<KEY>')}"
        ) from None
    ctype = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or not ctype.startswith("image/"):
        raise RuntimeError(
            f"Google Static Maps check failed: HTTP {resp.status_code} "
            f"{resp.text[:300].replace(api_key, '<KEY>')}"
        )


def new_report_map_google_static(api_key: str):
    """Production new_report_map() settings, with a Google Static Maps basemap."""
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
    script = _BASEMAP_JS % {
        "map_name": json.dumps(m.get_name()),
        "base_url": json.dumps(STATIC_MAPS_URL),
        "key": json.dumps(api_key),
        "img": STATIC_IMG_PX,
        "overlap": LOGO_OVERLAP_PX,
        "max_z": STATIC_MAX_ZOOM,
    }
    m.get_root().html.add_child(folium.Element(script))
    return m


def _copy_rendered_png(original):
    def _wrapped(html_path, png_path, *args, **kwargs):
        result = original(html_path, png_path, *args, **kwargs)
        if os.path.exists(png_path):
            MAPS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png_path, MAPS_DIR / os.path.basename(png_path))
        return result
    return _wrapped


@contextlib.contextmanager
def use_google_static_basemap(api_key: str):
    """
    Swap the basemap for the duration of the block. report_ppt_generator binds
    new_report_map / html_to_png at import time, and map_generator resolves
    new_report_map from its own module globals, so all of them are patched.
    """
    from tools.report_engine import map_generator, playwright_utils
    import report_ppt_generator

    def _patched_map():
        return new_report_map_google_static(api_key)

    copier = _copy_rendered_png(playwright_utils.html_to_png)
    patches = [
        (map_generator, "new_report_map", _patched_map),
        (report_ppt_generator, "new_report_map", _patched_map),
        (playwright_utils, "html_to_png", copier),
        (report_ppt_generator, "html_to_png", copier),
    ]
    originals = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, value in patches:
        setattr(mod, name, value)
    try:
        yield report_ppt_generator
    finally:
        for mod, name, value in originals:
            setattr(mod, name, value)


def run(project_id=DEFAULT_PROJECT_ID, country_code=DEFAULT_COUNTRY, session_ids=None, locked_bands=None):
    api_key = google_api_key()
    check_static_maps_key(api_key)
    print("[GoogleBasemap] Static Maps API key OK", flush=True)

    if MAPS_DIR.exists():
        shutil.rmtree(MAPS_DIR)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"Mobility_DT_Project_{project_id}_google_basemap.pptx"

    with use_google_static_basemap(api_key) as report_ppt_generator:
        report_ppt_generator.generate_ppt_for_project(
            project_id=project_id,
            session_ids=session_ids,
            country_code=country_code,
            region=country_code,
            output_path=str(output_path),
            locked_bands=locked_bands,
        )

    print(f"[GoogleBasemap] PPTX -> {output_path}", flush=True)
    print(f"[GoogleBasemap] Map PNGs -> {MAPS_DIR}", flush=True)
    return output_path


def test_ppt_google_basemap():
    output_path = run()
    assert output_path.exists()
    assert any(MAPS_DIR.glob("*.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT report with Google Static Maps basemap (test case)")
    parser.add_argument("--project-id", type=int, default=DEFAULT_PROJECT_ID)
    parser.add_argument("--country-code", type=str, default=DEFAULT_COUNTRY)
    parser.add_argument("--session-ids", type=str, default=None)
    parser.add_argument("--locked-bands", type=str, default=None)
    args = parser.parse_args()
    run(args.project_id, args.country_code, args.session_ids, args.locked_bands)
