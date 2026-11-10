"""Waste detection page: image (single/batch), video, webcam snapshot and live local camera."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from ui import page_header, get_config, get_db, get_detector, site_inputs, train_first

from src.utils.app_helpers import bgr_to_rgb, decode_image, df_to_csv_bytes, encode_png, mean_confidence, validate_coordinates
from src.utils.plots import class_counts_bar
from src.utils.report import waste_score

cfg = get_config()
inf = cfg["waste"]["inference"]
page_header("Live detection", ":material/radar:", "Detect and classify marine litter with YOLO11. Turn on underwater enhancement in the sidebar for murky, low-contrast footage.")

with st.sidebar:
    st.subheader("Detection settings", anchor=False)
    det_mode = st.segmented_control(
        "Detection mode", ["Standard", "High recall", "Maximum"], default="Standard", key="det_mode",
        help="Standard: one pass, real-time. High recall: also scans zoomed-in tiles, so it finds small and densely "
             "packed items (~10x slower). Maximum: high recall plus test-time augmentation (images only, a few seconds).",
    ) or "Standard"
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.15 if det_mode != "Standard" else float(inf["conf"]), 0.05,
                     key=f"conf_{det_mode}", help="Lower finds more items but adds more false alarms. Try 0.15 for crowded scenes.")
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, float(inf["iou"]), 0.05)
    enhance_on = st.toggle("Underwater enhancement", value=False, help="Dehaze + CLAHE before inference. Helps dark, murky footage; leave off for clear photos.")
    with st.expander("Enhancement steps"):
        t_dehaze = st.checkbox("Dark-channel dehaze", True)
        t_clahe = st.checkbox("CLAHE contrast", True)
        t_denoise = st.checkbox("Denoise", False)
        t_wb = st.checkbox("Gray-world white balance", bool(cfg["preprocessing"].get("white_balance", False)))

det = get_detector()
if det is None:
    train_first("waste")
    st.stop()
if det.is_demo:
    st.info("Using the generic COCO-pretrained demo model, which reports COCO labels such as *bottle* and *cup*. "
            "Train the 15-class marine-litter model with `python -m src.waste.train --mode demo`.", title="No trained waste model yet", icon=":material/info:")

MODE_KW = {"Standard": {"tiles": 1, "augment": False},
           "High recall": {"tiles": int(inf.get("high_recall_tiles", 3)), "augment": False},
           "Maximum": {"tiles": int(inf.get("high_recall_tiles", 3)), "augment": True}}
kw = dict(conf=conf, iou=iou, enhance_input=enhance_on, enhance_toggles={"dehaze": t_dehaze, "clahe": t_clahe, "denoise": t_denoise, "white_balance": t_wb}, **MODE_KW[det_mode])
# video and live camera need speed: cap them at "High recall" without test-time augmentation
kw_stream = {**kw, "augment": False}


def show_result(img, r, source: str) -> None:  # noqa: ANN001
    """Render before/after, annotated output, counts chart, table and downloads."""
    if enhance_on:
        a, b = st.columns(2)
        a.image(bgr_to_rgb(img), caption="Original", width="stretch", alt="Original image")
        b.image(bgr_to_rgb(r.input_image), caption="Enhanced (model input)", width="stretch", alt="Enhanced image")
    left, right = st.columns([3, 2])
    left.image(bgr_to_rgb(r.annotated), caption=f"{r.n} item(s) detected · {r.inference_ms:.0f} ms", width="stretch", alt="Image with detected waste boxes")
    with right:
        score = waste_score(r.n, r.coverage(), cfg)
        with st.container(horizontal=True):
            st.metric("Items", r.n, border=True)
            st.metric("Mean confidence", f"{mean_confidence(r.detections):.2f}" if r.n else "–", border=True)
            st.metric("Cleanliness", f"{score:.0f}/100", help="100 = no visible debris; penalises item count and area covered", border=True)
        if r.class_counts:
            st.plotly_chart(class_counts_bar(r.class_counts), alt="Bar chart of detected items per class", theme=None)
        else:
            st.success("No waste detected at this confidence threshold.", icon=":material/check_circle:")
    st.dataframe(r.detections.drop(columns=["timestamp"]), hide_index=True)
    stem = Path(source).stem or "image"
    with st.container(horizontal=True):
        st.download_button("Annotated image", encode_png(r.annotated), f"{stem}_detections.png", "image/png", icon=":material/download:")
        st.download_button("Detections CSV", df_to_csv_bytes(r.detections), f"{stem}_detections.csv", "text/csv", icon=":material/download:")
        with st.popover("Log to history", icon=":material/save:"):
            with st.form(f"log_{stem}", border=False):
                name, lat, lon = site_inputs(f"log_{stem}")
                if st.form_submit_button("Save record", type="primary"):
                    try:
                        lat, lon = validate_coordinates(lat, lon)
                        get_db().log_waste(r.n, r.class_counts, mean_confidence(r.detections), r.coverage(), score, name or None, lat, lon, source, enhance_on)
                        st.success("Saved to history.")
                    except ValueError as exc:
                        st.error(str(exc))


mode = st.segmented_control("Input", ["Image", "Video", "Webcam snapshot", "Live camera"], default="Image", key="waste_mode",
                            format_func=lambda m: {"Image": ":material/image: Image", "Video": ":material/movie: Video", "Webcam snapshot": ":material/photo_camera: Webcam snapshot", "Live camera": ":material/videocam: Live camera"}[m])

if mode == "Image":
    files = st.file_uploader("Upload one or more underwater images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
    if not files:
        st.caption("Tip: after the data step, sample images are in `data/processed/yolo/images/test/`.")
    else:
        current = files[0] if len(files) == 1 else next(f for f in files if f.name == st.selectbox("Image to inspect", [f.name for f in files]))
        try:
            img = decode_image(current.getvalue())
            with st.spinner("Detecting..."):
                res = det.predict(img, source=current.name, **kw)
            show_result(img, res, current.name)
        except ValueError as exc:
            st.error(f"Could not process {current.name}: {exc}", icon=":material/error:")
        if len(files) > 1 and st.button(f"Run batch on all {len(files)} images", icon=":material/stacks:"):
            prog = st.progress(0.0, "Batch inference...")
            frames = []
            for i, f in enumerate(files):
                try:
                    frames.append(det.predict(decode_image(f.getvalue()), source=f.name, **kw).detections)
                except ValueError as exc:
                    st.warning(f"{f.name}: {exc}")
                prog.progress((i + 1) / len(files), f"{i + 1}/{len(files)}")
            batch = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["source", "class_name"])
            st.dataframe(batch.groupby(["source", "class_name"]).size().unstack(fill_value=0) if not batch.empty else batch)
            st.download_button("Batch detections CSV", df_to_csv_bytes(batch), "batch_detections.csv", "text/csv", icon=":material/download:")

elif mode == "Video":
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
    with st.container(horizontal=True):
        stride = st.number_input("Process every Nth frame", 1, 30, 2)
        max_frames = st.number_input("Max frames to process", 10, 5000, int(inf["max_video_frames"]))
    if vid is not None and st.button("Run detection on video", type="primary", icon=":material/play_arrow:"):
        tmpdir = Path(tempfile.mkdtemp(prefix="owd_"))
        src = tmpdir / Path(vid.name).name
        src.write_bytes(vid.getvalue())
        prog = st.progress(0.0, "Processing video...")
        try:
            t0 = time.time()
            df, out, samples = det.predict_video(src, tmpdir / f"{src.stem}_detections.webm", stride=int(stride), max_frames=int(max_frames),
                                                 progress=lambda p: prog.progress(p, f"{p:.0%}"), **kw_stream)
            prog.progress(1.0, f"Done in {time.time() - t0:.1f}s")
            if out and out.exists():
                data = out.read_bytes()
                mime = "video/webm" if out.suffix == ".webm" else "video/mp4"
                st.video(data, format=mime)
                st.download_button("Annotated video", data, out.name, mime, icon=":material/download:")
            if samples:
                st.image([bgr_to_rgb(s) for s in samples], width=300, caption=[f"Sample frame {i + 1}" for i in range(len(samples))], alt=[f"Annotated sample frame {i + 1}" for i in range(len(samples))])
            st.metric("Total detections", len(df), border=True)
            if not df.empty:
                st.plotly_chart(class_counts_bar(df["class_name"].value_counts().to_dict(), "Detections across processed frames"), theme=None, alt="Detections per class across the video")
                st.line_chart(df.groupby("frame").size().rename("Items per frame"), x_label="Frame", y_label="Items")
            st.download_button("Detections CSV", df_to_csv_bytes(df), f"{src.stem}_detections.csv", "text/csv", icon=":material/download:")
        except (ValueError, RuntimeError) as exc:
            st.error(f"Video processing failed: {exc}", icon=":material/error:")

elif mode == "Webcam snapshot":
    snap = st.camera_input("Take a snapshot (works in any browser)")
    if snap is not None:
        try:
            img = decode_image(snap.getvalue())
            show_result(img, det.predict(img, source="webcam_snapshot", **kw), "webcam_snapshot")
        except ValueError as exc:
            st.error(str(exc))

elif mode == "Live camera":
    st.caption("Streams from a camera attached to the machine running this app, with GPU FP16 inference. Switch off to stop.")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        cam_idx = int(st.number_input("Camera index", 0, 9, 0))
        live = st.toggle("Start live detection", key="live_toggle")
    if not live:
        cap = st.session_state.pop("live_cap", None)
        if cap is not None:
            cap.release()
    else:
        cap = st.session_state.get("live_cap")
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
            st.session_state["live_cap"] = cap
        if not cap.isOpened():
            st.error(f"Cannot open camera {cam_idx}. Check that a webcam is connected and not used by another app.", icon=":material/videocam_off:")
        else:
            @st.fragment(run_every=0.1)
            def live_frame() -> None:
                c = st.session_state.get("live_cap")
                ok, frame = (c.read() if c is not None else (False, None))
                if not ok:
                    st.warning("The camera is not delivering frames.")
                    return
                r = det.predict(frame, source=f"webcam{cam_idx}", **kw_stream)
                st.image(bgr_to_rgb(r.annotated), width="stretch", alt="Live camera frame with detections")
                st.caption(f"**{r.n}** item(s) · {r.inference_ms:.0f} ms/frame · " + (", ".join(f"{k}: {v}" for k, v in r.class_counts.items()) or "no waste"))

            live_frame()
