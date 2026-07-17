"""
Google Maps "Map Tiles API" basemap integration (test-case only;
tools/report_engine is NOT modified). Swaps the CartoDB Voyager tiles
every map in this report currently uses for real Google-rendered
imagery, per direction received (the free tile provider's colors were
not good enough).

Uses Google's OFFICIALLY SUPPORTED Map Tiles API (tile.googleapis.com),
NOT the unofficial mt0-mt3.google.com/vt endpoints some tutorials/scripts
use -- those aren't a supported product, and using them outside Google's
own Maps JavaScript SDK is against Google's Terms of Service. The Map
Tiles API is a real, documented, paid REST tile service that plugs into
Leaflet as a standard XYZ tile layer once you have a session token:
  https://developers.google.com/maps/documentation/tile/2d-tiles-overview

This is a deliberately different (smaller) integration than the
Static-Maps-API-plus-OpenCV approach in the reference script we were
given: that approach downloads one flat image per request and draws
everything onto it manually. Using the Tile API instead means every
existing Folium/Leaflet marker, legend, polygon-overlay and grid-cell
drawing routine in this report keeps working completely unchanged --
only the basemap image source underneath them changes.

Flow: POST /v1/createSession ONCE per script run (session tokens last
about 2 weeks per Google's docs) to get a session token, then every tile
is a plain authenticated GET using that token -- so we cache one session
per map_type and reuse it for every map in the report, rather than
creating a new (billed) session per map.
"""
import contextlib
import time

import requests

_SESSION_CACHE: dict[str, tuple[str, float]] = {}  # map_type -> (session_token, expiry_epoch)


def get_google_tile_session(api_key: str, map_type: str = "roadmap") -> str:
    """Cached session token for `map_type`; creates a new one only if
    none is cached yet or the cached one is close to expiring."""
    cached = _SESSION_CACHE.get(map_type)
    if cached and cached[1] > time.time() + 60:
        return cached[0]

    resp = requests.post(
        f"https://tile.googleapis.com/v1/createSession?key={api_key}",
        json={"mapType": map_type, "language": "en-US", "region": "US"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    session_token = data["session"]
    expiry = float(data.get("expiry", time.time() + 3600))
    _SESSION_CACHE[map_type] = (session_token, expiry)
    return session_token


def new_report_map_google(api_key: str, map_type: str = "roadmap"):
    """Drop-in replacement for production's new_report_map()
    (tools/report_engine/map_generator.py, NOT modified) -- same
    folium.Map settings (fractional zoom, etc.), but with a Google Map
    Tiles API layer instead of CartoDB Voyager."""
    import folium
    from tools.report_engine.map_generator import REPORT_MAP_MAX_ZOOM

    session_token = get_google_tile_session(api_key, map_type)
    tile_url = (
        "https://tile.googleapis.com/v1/2dtiles/{z}/{x}/{y}"
        f"?session={session_token}&key={api_key}"
    )
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
        tiles=tile_url,
        attr="© Google",
        name="Google Maps",
        overlay=False,
        control=False,
        max_zoom=REPORT_MAP_MAX_ZOOM,
    ).add_to(m)
    return m


@contextlib.contextmanager
def use_google_tiles(api_key: str, map_type: str = "roadmap"):
    """
    Monkeypatches every known reference to production's new_report_map so
    EVERY map in this report -- both our own test-case map functions
    (which import new_report_map locally, inside each function, so they
    automatically pick up whatever map_generator.new_report_map currently
    is) and the production raw-point functions we call unmodified
    (generate_kpi_map, generate_categorical_kpi_map,
    generate_base_route_map, ... -- these resolve new_report_map via
    map_generator's own module globals at call time, the same mechanism
    the existing _polygon_aware_fit_bounds patch in new_report_sections.py
    already relies on) -- pick up Google tiles for the duration of this
    block. grid_rsrp_map_test.py is the one exception: it imports
    new_report_map at MODULE level, binding its own separate reference,
    so it needs patching directly too. Restores the originals afterward;
    tools/report_engine is never modified on disk.
    """
    from tools.report_engine import map_generator
    from tests.new_pdf_report import grid_rsrp_map_test

    def _patched():
        return new_report_map_google(api_key, map_type)

    original_mg = map_generator.new_report_map
    original_grid = grid_rsrp_map_test.new_report_map
    map_generator.new_report_map = _patched
    grid_rsrp_map_test.new_report_map = _patched
    try:
        yield
    finally:
        map_generator.new_report_map = original_mg
        grid_rsrp_map_test.new_report_map = original_grid


def new_report_map_gray_clipped(polygon_wkt: str):
    """
    Drop-in replacement for production's new_report_map() (map_generator.py,
    NOT modified) -- identical folium.Map settings and the normal CartoDB
    Voyager base tiles everywhere, PLUS a second "CartoDB positron" (light,
    muted, near-grayscale, no colored roads) tile layer stacked on top and
    clipped with a CSS clip-path to ONLY the polygon's own interior.

    Voyager renders roads in yellow, which is also one of our own KPI
    value-range colors (e.g. RSRP "-90 to -80" is yellow -- see
    kpi_config.rsrp_colour_manual), so on a grid map the basemap's own
    roads visually blend with the grid legend's colors. Positron has no
    such color to collide with -- but per direction received, the gray
    treatment should apply ONLY inside the polygon (the actual analysis
    area), not the whole screenshot; anything outside the polygon (e.g.
    padding/context area from fit_bounds_including_polygon) stays on the
    normal colorful Voyager tiles.

    Implementation: a custom Leaflet pane ("grayClipPane") holding the
    Positron tile layer is created above the base tile pane, then its
    clip-path is computed from the polygon's own vertices projected to
    on-screen layer points via map.latLngToLayerPoint -- the same
    coordinate space Leaflet itself uses to position/transform panes, so
    the clip stays aligned with the basemap under panning/zooming. The
    clip is (re)computed on 'moveend' (which fires after fit_bounds'
    viewport change settles) so it is correct by the time playwright's
    tile-loaded wait in html_to_png() lets the screenshot proceed.

    Interim fix, per direction received, until the Google Maps Tile API
    integration (google_tiles.py, use_google_tiles()) is unblocked on the
    API-key/billing side.
    """
    import folium
    from shapely.wkt import loads as load_wkt

    from tools.report_engine.map_generator import REPORT_MAP_MAX_ZOOM

    m = folium.Map(
        tiles="CartoDB Voyager",
        zoom_control=True,
        control_scale=False,
        prefer_canvas=True,
        max_zoom=REPORT_MAP_MAX_ZOOM,
        zoomSnap=0,
        zoomDelta=0.25,
    )

    polygon = load_wkt(polygon_wkt)
    latlon = [[c[1], c[0]] for c in polygon.exterior.coords]

    script = f"""
    (function() {{
        function initGrayClip(retries) {{
            var mapEl = document.querySelector('.folium-map');
            var map = mapEl && mapEl.id && window[mapEl.id];
            if (!map) {{
                if (retries > 0) {{ setTimeout(function() {{ initGrayClip(retries - 1); }}, 100); }}
                return;
            }}

            var grayPane = map.createPane('grayClipPane');
            grayPane.style.zIndex = 350;

            var grayLayer = L.tileLayer(
                'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png',
                {{ pane: 'grayClipPane', maxZoom: {REPORT_MAP_MAX_ZOOM}, attribution: '' }}
            ).addTo(map);

            var polygonLatLngs = {latlon!r};

            function updateClip() {{
                var points = polygonLatLngs.map(function(ll) {{
                    var p = map.latLngToLayerPoint(L.latLng(ll[0], ll[1]));
                    return p.x + 'px ' + p.y + 'px';
                }});
                var clip = 'polygon(' + points.join(',') + ')';
                grayPane.style.clipPath = clip;
                grayPane.style.webkitClipPath = clip;
            }}

            map.on('moveend', updateClip);
            grayLayer.on('load', updateClip);
            updateClip();
        }}
        initGrayClip(50);
    }})();
    """
    m.get_root().script.add_child(folium.Element(script))
    return m


@contextlib.contextmanager
def use_gray_basemap(polygon_wkt: str):
    """
    Monkeypatches new_report_map to the polygon-clipped grayscale variant
    (see new_report_map_gray_clipped()) for the duration of this block --
    same patch targets/mechanism as use_google_tiles() above. Scoped
    narrowly to GRID maps only (wrap just the grid-rendering call sites,
    e.g. in _render_kpi_map_per_technology's `if use_grid:` branch and the
    poor-region grid branch), which are also exactly the call sites that
    always have a polygon_wkt available. Raw-point (no-polygon) maps keep
    the original CartoDB Voyager tiles unchanged, and even within a grid
    map only the polygon's own interior is grayed out.
    """
    from tools.report_engine import map_generator
    from tests.new_pdf_report import grid_rsrp_map_test

    def _patched():
        return new_report_map_gray_clipped(polygon_wkt)

    original_mg = map_generator.new_report_map
    original_grid = grid_rsrp_map_test.new_report_map
    map_generator.new_report_map = _patched
    grid_rsrp_map_test.new_report_map = _patched
    try:
        yield
    finally:
        map_generator.new_report_map = original_mg
        grid_rsrp_map_test.new_report_map = original_grid
