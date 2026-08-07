"""
Geo Data Zip Inspector

Run with:
    ML/venv/Scripts/streamlit.exe run ML/tests/geodata/dashboard.py

Purpose: before a client's "Maps Data" zip is used for saved-polygon /
OSM geo-enrichment, inspect what's actually inside it, check whether nested
archives are complete (not truncated by a bad upload/download), and preview
a small snapshot on a map - all without needing to fully extract huge
(300MB-1GB+) files up front.
"""
import shutil
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import geo_inspector as gi

st.set_page_config(page_title="Geo Data Zip Inspector", layout="wide")

DEFAULT_ZIP = str(Path(__file__).parents[2] / "data" / "Maps Data.zip")
SCRATCH_DIR = Path(__file__).parent / "_scratch"

st.title("🗺️ Geo Data Zip Inspector")
st.caption(
    "Peek inside a client's Maps Data zip, check nested archives for corruption, "
    "and preview a small snapshot before running polygon-save / OSM geo enrichment."
)

with st.sidebar:
    st.header("Source")
    zip_path = st.text_input("Zip file path", value=DEFAULT_ZIP)
    scan = st.button("Scan zip", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Scratch dir for staged/extracted files:\n`{SCRATCH_DIR}`")
    if st.button("Clear scratch dir", use_container_width=True):
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
        st.session_state.clear()
        st.success("Cleared.")

if scan:
    st.session_state["zip_path"] = zip_path
    st.session_state.pop("entries", None)

active_zip = st.session_state.get("zip_path")

if not active_zip:
    st.info("Enter a zip path in the sidebar and click **Scan zip** to begin.")
    st.stop()

if not Path(active_zip).exists():
    st.error(f"File not found: {active_zip}")
    st.stop()

# ---------------------------------------------------------------- overview --
zip_size = Path(active_zip).stat().st_size
st.subheader("1. Zip overview")
st.write(f"**{Path(active_zip).name}** — {gi.human_size(zip_size)} on disk")

if "entries" not in st.session_state:
    with st.spinner("Reading zip directory..."):
        st.session_state["entries"] = gi.scan_zip(active_zip)
entries = st.session_state["entries"]

df = gi.entries_to_dataframe(entries)
st.dataframe(df[["name", "type", "size", "compressed"]], use_container_width=True, hide_index=True)

file_entries = [e for e in entries if not e.is_dir]
archive_entries = [e for e in file_entries if e.ext in gi.ARCHIVE_EXTS]
geo_entries = [e for e in file_entries if e.ext in gi.GEO_FILE_EXTS]

if not archive_entries and not geo_entries:
    st.warning("No nested archives or recognizable geo files found at the top level of this zip.")
    st.stop()

# --------------------------------------------------------- nested archives --
if archive_entries:
    st.subheader("2. Nested archive integrity check")
    st.caption(
        "Reads only the first 32 bytes of each nested archive (streamed, no disk write) "
        "to verify the file wasn't cut off mid-transfer, before we bother staging it."
    )

    for entry in archive_entries:
        with st.container(border=True):
            st.markdown(f"**{entry.name}**  ·  {gi.human_size(entry.size)}")

            if entry.ext == ".7z":
                check_key = f"check::{entry.name}"
                if check_key not in st.session_state:
                    with st.spinner("Checking header..."):
                        st.session_state[check_key] = gi.precheck_7z_integrity_in_zip(
                            active_zip, entry.name
                        )
                result = st.session_state[check_key]

                if not result.signature_ok:
                    st.error(f"⚠️ {result.note}")
                elif result.is_complete:
                    st.success("✅ Archive appears complete — header and all declared data present.")
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Present on disk", gi.human_size(result.actual_size))
                    c2.metric("Expected size", gi.human_size(result.expected_min_size))
                    c3.metric("Missing", f"{result.missing_pct:.1f}%")
                    st.progress(result.actual_size / result.expected_min_size)
                    st.error(f"🚫 **Truncated / incomplete archive.** {result.note}")
                    st.markdown(
                        "**This is very likely why saved-polygon / OSM geo-enrichment is failing "
                        "on this dataset.** The 7z file's index (which lists every file inside) is "
                        "stored at the end of the archive, and that part never arrived. "
                        "Nothing can be listed or extracted from it in this state — "
                        "re-copy/re-download the source file from the client and re-check here."
                    )
                    continue  # don't offer staging for a file we know is broken

            elif entry.ext == ".zip":
                st.info("Nested .zip — integrity pre-check not implemented, will attempt to stage directly.")
            else:
                st.info(f"'{entry.ext}' archives aren't specially handled yet — will attempt to stage directly.")

            stage_key = f"staged::{entry.name}"
            if st.button(f"Stage & list contents", key=f"btn_{entry.name}"):
                dest_dir = SCRATCH_DIR / "staged"
                progress = st.progress(0.0, text="Staging (copying out of zip)...")

                def _cb(written, total, _p=progress):
                    _p.progress(min(written / total, 1.0), text=f"Staging... {gi.human_size(written)} / {gi.human_size(total)}")

                staged_path = gi.stage_entry(active_zip, entry.name, str(dest_dir), progress_cb=_cb)
                progress.empty()
                st.session_state[stage_key] = str(staged_path)

            if stage_key in st.session_state:
                staged_path = st.session_state[stage_key]
                st.caption(f"Staged at `{staged_path}`")
                try:
                    contents_df = gi.list_7z_contents(staged_path) if entry.ext == ".7z" else None
                except Exception as e:
                    st.error(f"Failed to list contents: {e}")
                    contents_df = None

                if contents_df is not None and not contents_df.empty:
                    st.write(f"{len(contents_df)} files inside:")
                    st.dataframe(contents_df, use_container_width=True, hide_index=True)

                    shp_files = [p for p in contents_df["Path"] if p.lower().endswith(".shp")]
                    if shp_files:
                        chosen = st.selectbox("Preview a shapefile layer:", shp_files, key=f"sel_{entry.name}")
                        if st.button("Extract snapshot & preview", key=f"prev_{entry.name}"):
                            stem = Path(chosen).with_suffix("")
                            sidecars = [
                                p for p in contents_df["Path"]
                                if Path(p).with_suffix("") == stem and Path(p).suffix.lower() in gi.SHAPEFILE_SIDECAR_EXTS
                            ]
                            snap_dir = SCRATCH_DIR / "snapshot"
                            with st.spinner(f"Extracting {len(sidecars)} sidecar file(s)..."):
                                extracted = gi.extract_from_7z(staged_path, sidecars, str(snap_dir))
                            shp_local = next(p for p in extracted if p.suffix.lower() == ".shp")
                            st.session_state[f"preview_path::{entry.name}"] = str(shp_local)

# ------------------------------------------------------- direct geo files --
if geo_entries:
    st.subheader("2b. Geo files at the top level of the zip")
    for entry in geo_entries:
        with st.container(border=True):
            st.markdown(f"**{entry.name}**  ·  {gi.human_size(entry.size)}")
            if st.button("Extract & preview", key=f"direct_{entry.name}"):
                dest_dir = SCRATCH_DIR / "snapshot"
                staged = gi.stage_entry(active_zip, entry.name, str(dest_dir))
                st.session_state[f"preview_path::{entry.name}"] = str(staged)

# ------------------------------------------------------------------ preview --
preview_keys = [k for k in st.session_state if k.startswith("preview_path::")]
if preview_keys:
    st.subheader("3. Snapshot preview")
    for key in preview_keys:
        path = st.session_state[key]
        st.markdown(f"**{Path(path).name}**")
        try:
            with st.spinner("Loading with geopandas..."):
                preview = gi.load_geo_preview(path)
        except Exception as e:
            st.error(f"Could not load {path}: {e}")
            continue

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Driver", preview.driver)
        c2.metric("CRS", preview.crs or "unknown")
        c3.metric("Feature count", f"{preview.feature_count:,}")
        c4.metric("Geometry type(s)", ", ".join(preview.geometry_types) or "n/a")

        if preview.bounds:
            st.caption(f"Bounds (minx, miny, maxx, maxy): {tuple(round(b, 4) for b in preview.bounds)}")

        st.write("Attribute columns:", ", ".join(preview.columns))
        st.dataframe(preview.sample, use_container_width=True, hide_index=True)

        if preview.sample_gdf is not None and len(preview.sample_gdf):
            import folium
            from streamlit_folium import st_folium

            gdf_map = preview.sample_gdf.to_crs(4326) if preview.crs else preview.sample_gdf
            minx, miny, maxx, maxy = gdf_map.total_bounds
            m = folium.Map()
            m.fit_bounds([[miny, minx], [maxy, maxx]])
            folium.GeoJson(gdf_map.__geo_interface__).add_to(m)
            st_folium(m, use_container_width=True, height=500, key=f"map_{key}")
