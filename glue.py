"""Python side of the web spike, run inside Pyodide.

The page passes in what the browser has (MediaPipe's JavaScript face landmarks and the
camera frame's RGBA pixels), and these functions call the same ``snellen`` code the
desktop app uses. They return plain dicts that convert cleanly to JavaScript objects.
"""

import random
import time

import cv2
import numpy as np

from snellen.acuity import AcuityTest
from snellen.card import find_card
from snellen.face import eye_spans_px, pupils_px
from snellen.geometry import (
    LANDOLT_C_RADII,
    Direction,
    feature_width_mm,
    landolt_c_gap,
    optotype_height_mm,
    optotype_height_px,
    pixel_distance_px,
    tumbling_e_cells,
)
from snellen.optotypes import available_optotypes


def measure(landmarks, width_px: int, height_px: int, rgba=None) -> dict:
    """Pupils, one-eye spans and (if a frame is given) the card, for one camera frame.

    Parameters
    ----------
    landmarks : JS array or None
        One face's landmarks from the JavaScript Face Landmarker (objects with ``x``
        and ``y`` normalised to 0..1), or ``None`` if no face was found. From
        JavaScript, pass ``undefined`` (which arrives as ``None``), not ``null`` (which
        arrives as a ``JsNull`` object).
    width_px, height_px : int
        Frame size.
    rgba : JS Uint8ClampedArray, optional
        The frame's pixels from ``getImageData``, to look for the card (``undefined``
        from JavaScript to skip it).

    Returns
    -------
    dict
        ``pupils`` (two ``[x, y]``), ``ipd_px``, ``spans`` (per-eye nose and iris px),
        ``card`` (``width_px`` and ``corners``), ``card_mm_ipd`` (the pupil distance the
        card implies, in mm) and timings ``copy_ms`` and ``card_ms``. Missing things
        are ``None``.
    """
    out: dict = {"pupils": None, "ipd_px": None, "spans": None, "card": None}
    pupils = None
    if landmarks is not None:
        pupils = pupils_px(landmarks, width_px, height_px)
        spans = eye_spans_px(landmarks, width_px, height_px)
        if pupils is not None:
            out["pupils"] = [list(p) for p in pupils]
            out["ipd_px"] = pixel_distance_px(*pupils)
        if spans is not None:
            out["spans"] = {
                "left_nose_px": spans.left_nose_px,
                "right_nose_px": spans.right_nose_px,
                "left_iris_px": spans.left_iris_px,
                "right_iris_px": spans.right_iris_px,
            }
    if rgba is not None:
        start = time.perf_counter()
        frame = np.frombuffer(rgba.to_py(), np.uint8).reshape(height_px, width_px, 4)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        copied = time.perf_counter()
        card = find_card(bgr, pupils)
        done = time.perf_counter()
        out["copy_ms"] = (copied - start) * 1000
        out["card_ms"] = (done - copied) * 1000
        if card is not None:
            out["card"] = {"width_px": card.width_px, "corners": [list(c) for c in card.corners_px]}
            if out["ipd_px"]:
                out["card_mm_ipd"] = feature_width_mm(out["ipd_px"], card.width_px)
    return out


def chart_names() -> list[str]:
    """The chart types, in the order the desktop app offers them."""
    return [o.name for o in available_optotypes()]


def chart_line(
    name: str, denominator: int, distance_mm: float, px_per_mm: float, seed: int
) -> dict:
    """One chart line to draw: size and symbols, from the same code as the desktop app.

    Returns
    -------
    dict
        ``height_px`` (in CSS pixels), ``height_mm`` (for the ruler check), ``kind``
        (``"letters"``, ``"e"`` or ``"c"``) and ``shapes``: per optotype, either the
        letter, or the E's rectangles / the C's gap on the 5 x 5 grid.
    """
    optotype = next(o for o in available_optotypes() if o.name == name)
    test = AcuityTest(random.Random(seed), symbols=optotype.symbols)
    shapes = []
    for symbol in test.symbols:
        if not isinstance(symbol, Direction):
            shapes.append({"letter": symbol})
        elif optotype.name == "Tumbling E":
            shapes.append({"cells": [list(c) for c in tumbling_e_cells(symbol)]})
        else:
            shapes.append({"gap": list(landolt_c_gap(symbol)), "radii": list(LANDOLT_C_RADII)})
    kind = {"Snellen letters": "letters", "Tumbling E": "e"}.get(optotype.name, "c")
    return {
        "height_px": optotype_height_px(distance_mm, denominator, px_per_mm),
        "height_mm": optotype_height_mm(distance_mm, denominator),
        "kind": kind,
        "shapes": shapes,
    }
