import argparse
import sys
import os
import json
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw
from dataclasses import dataclass
from collections import deque
import time
import pydobot


device = pydobot.Dobot(port="COM5")


# ============================================================================
# CORE HELPERS
# ============================================================================

def make_dir_if_needed(path: str):
    dirn = os.path.dirname(path)
    if dirn and not os.path.exists(dirn):
        os.makedirs(dirn, exist_ok=True)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def binarize_white_foreground(gray: np.ndarray) -> np.ndarray:
    """Threshold so white maze remains white (255) and black background is 0."""
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (th > 0).mean() > 0.9:  # if almost everything turned white, invert
        th = 255 - th
    return th


def morph_cleanup(bin_img: np.ndarray) -> np.ndarray:
    """Light open + close to reduce specks and bridge small gaps."""
    opened = cv2.morphologyEx(
        bin_img,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    return closed


def approx_to_quads(cnt: np.ndarray, max_iter: int = 25) -> Optional[np.ndarray]:
    """Try to approximate the contour to exactly 4 points by increasing epsilon."""
    peri = cv2.arcLength(cnt, True)
    for frac in np.linspace(0.01, 0.06, max_iter):
        approx = cv2.approxPolyDP(cnt, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def find_largest_quad(bin_img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Find the largest-area quadrilateral among contours.
    Fallback to minAreaRect on the largest contour if needed.
    """
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("No contours found. Check thresholding/lighting.")

    h, w = bin_img.shape[:2]
    min_area = max(1000.0, 0.001 * (h * w))
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        candidates = [max(contours, key=cv2.contourArea)]

    best_quad, best_area = None, -1.0
    for c in candidates:
        quad = approx_to_quads(c)
        if quad is not None:
            area = cv2.contourArea(quad.astype(np.int32))
            if area > best_area:
                best_area = area
                best_quad = quad

    if best_quad is not None:
        return order_corners(best_quad), "largest_quadrilateral"

    # Fallback: oriented bounding box of the largest candidate
    largest = max(candidates, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_corners(box), "minAreaRect_fallback"


def expand_corners(corners: np.ndarray, expand_px: float) -> np.ndarray:
    """
    Move each corner outward from the polygon centroid by expand_px pixels.
    Positive = outward, negative = inward.
    """
    if abs(expand_px) < 1e-9:
        return corners.astype(np.float32)

    center = corners.mean(axis=0)
    expanded = []
    for p in corners:
        v = p - center
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            expanded.append(p)
        else:
            scale = (norm + expand_px) / norm
            expanded.append(center + v * scale)
    return order_corners(np.array(expanded, dtype=np.float32))


def draw_overlay(base_bgr: np.ndarray, corners: np.ndarray, color_poly=(255, 0, 0)) -> np.ndarray:
    overlay = base_bgr.copy()
    c = corners.astype(int)
    labels = ["TL", "TR", "BR", "BL"]
    cv2.polylines(overlay, [c.reshape(-1, 1, 2)], True, color_poly, 3)
    for i, p in enumerate(c):
        cv2.circle(overlay, tuple(p), 8, (0, 255, 0), -1)
        cv2.putText(
            overlay, labels[i], tuple(p + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )
    return overlay


def detect_corners_blackbg(img_bgr: np.ndarray, resize_width: Optional[int]) -> Tuple[np.ndarray, str, float]:
    """
    Detect corners at processing resolution; caller is responsible for upscaling to original.
    Returns (corners_at_processing_scale, method, scale).
    """
    if resize_width is not None and resize_width > 0 and img_bgr.shape[1] > resize_width:
        scale = resize_width / img_bgr.shape[1]
        img_proc = cv2.resize(
            img_bgr,
            (int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)),
            cv2.INTER_AREA
        )
    else:
        scale = 1.0
        img_proc = img_bgr

    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = binarize_white_foreground(gray)
    bin_img = morph_cleanup(bin_img)

    corners, method = find_largest_quad(bin_img)
    return corners, method, scale


# ============================================================================
# CAMERA UTILITIES
# ============================================================================

def open_camera(cam_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(
        cam_index, cv2.CAP_DSHOW
    ) if os.name == "nt" else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {cam_index}.")
    return cap


def camera_loop_and_capture(cam_index: int, window_name: str = "Camera - SPACE/c capture, q to quit") -> Optional[np.ndarray]:
    cap = open_camera(cam_index)
    last_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read from camera.")
                break
            last_frame = frame
            disp = frame.copy()
            h, w = disp.shape[:2]
            cv2.putText(
                disp, "SPACE/c: capture  |  q/ESC: quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                last_frame = None
                break
            if key in (32, ord('c')):
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
    return last_frame


# ============================================================================
# PIPELINE & SAVING
# ============================================================================

def run_detection_and_save(frame: np.ndarray, args):
    # Detect (possibly at reduced width), then map corners back to ORIGINAL pixels
    corners_proc, method, scale = detect_corners_blackbg(
        frame, None if args.width == 0 else args.width
    )
    corners = corners_proc / scale if scale != 1.0 else corners_proc.copy()
    corners = expand_corners(corners, args.expand)

    # Draw overlay
    overlay = draw_overlay(frame, corners)

    # Decide filenames
    raw_path = args.save_raw or "camera_capture.png"
    out_overlay_path = args.out
    json_path = args.json_out or "camera_capture_corners.json"

    # Save raw frame (so JSON 'input' references a real file)
    ok = cv2.imwrite(raw_path, frame)
    if ok:
        print(f"Raw frame saved to: {raw_path}")
    else:
        print("Warning: failed to save raw frame.")

    # Save overlay
    ok = cv2.imwrite(out_overlay_path, overlay)
    if not ok:
        print("Warning: failed to save overlay image.")
    else:
        print(f"Overlay saved to: {out_overlay_path}")

    # Save corners JSON
    corners_list = corners.tolist()
    data = {
        "input": os.path.basename(raw_path),
        "method": method,
        "width_param": args.width,
        "expand_param": args.expand,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "corners": {
            "TL": corners_list[0],
            "TR": corners_list[1],
            "BR": corners_list[2],
            "BL": corners_list[3]
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Corners JSON saved to: {json_path}")

    # Show result and allow quick re-save/retake
    show = overlay.copy()
    h, w = show.shape[:2]
    cv2.putText(
        show, "Press 's' to save again, 'r' to retake, 'q' to quit",
        (10, max(24, h - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA
    )
    cv2.imshow("Detected corners (overlay)", show)
    return raw_path, out_overlay_path, json_path


def capture_image_main():
    ap = argparse.ArgumentParser(
        description="Camera-only maze corners detector."
    )
    ap.add_argument(
        "--cam-index", type=int, default=0,
        help="Camera index to open (default: 0)."
    )
    ap.add_argument(
        "--out", default="part_2_maze_solution/maze_corners_overlay.png",
        help="Path to save overlay visualization"
    )
    ap.add_argument(
        "--width", type=int, default=1200,
        help="Processing resize width (0 to disable)"
    )
    ap.add_argument(
        "--expand", type=float, default=10.0,
        help="Outward expansion in pixels (negative shrinks inward)"
    )
    ap.add_argument(
        "--json-out", type=str, default="part_2_maze_solution/camera_capture_corners.json",
        help="Corners JSON output path (default: camera_capture_corners.json)"
    )
    ap.add_argument(
        "--save-raw", type=str, default="part_2_maze_solution/camera_capture.png",
        help="Optional path to save the raw captured frame (default: camera_capture.png)"
    )
    args = ap.parse_args()

    # Live preview -> capture
    frame = camera_loop_and_capture(args.cam_index)
    if frame is None:
        print("No frame captured. Exiting.")
        sys.exit(0)

    # Detect & save once
    raw_path, overlay_path, json_path = run_detection_and_save(frame, args)

    # Post-capture loop (retake/resave/quit)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            print("Saved (overlay & JSON already written).")
        elif key == ord('r'):
            cv2.destroyAllWindows()
            frame = camera_loop_and_capture(args.cam_index)
            if frame is None:
                print("No frame captured. Exiting.")
                break
            raw_path, overlay_path, json_path = run_detection_and_save(frame, args)
        else:
            continue

    cv2.destroyAllWindows()


# ============================================================================
# MAZE WARP FROM JSON
# ============================================================================

def parse_target_size(s: str) -> Tuple[int, int]:
    try:
        w_str, h_str = s.lower().split("x")
        w, h = int(w_str), int(h_str)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        raise ValueError("Invalid --target-size. Use WIDTHxHEIGHT (e.g., 1200x800).")


def infer_size_from_quad(corners: np.ndarray) -> Tuple[int, int]:
    tl, tr, br, bl = corners
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    W = int(round((width_top + width_bottom) / 2.0))
    H = int(round((height_left + height_right) / 2.0))
    return max(W, 1), max(H, 1)


def warp_perspective(img: np.ndarray, corners: np.ndarray, out_size: Tuple[int, int], pad: int = 0) -> np.ndarray:
    W, H = out_size
    src = corners.astype(np.float32)
    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
    if pad > 0:
        warped = cv2.copyMakeBorder(
            warped, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return warped


def read_corners_from_json(json_path: str) -> Tuple[np.ndarray, str]:
    """
    Returns (corners 4x2 float32 in TL,TR,BR,BL order, image_path_from_json_or_empty)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    img_from_json = ""
    if isinstance(data, dict):
        # Try nested dict under "corners"
        if "corners" in data and isinstance(data["corners"], dict):
            try:
                tl = data["corners"]["TL"]
                tr = data["corners"]["TR"]
                br = data["corners"]["BR"]
                bl = data["corners"]["BL"]
                pts = np.array([tl, tr, br, bl], dtype=np.float32)
            except KeyError as e:
                raise ValueError(f"Missing key in corners JSON: {e}")
        # Or root-level TL/TR/BR/BL
        elif all(k in data for k in ("TL", "TR", "BR", "BL")):
            pts = np.array(
                [data["TL"], data["TR"], data["BR"], data["BL"]], dtype=np.float32
            )
        # Or root-level list
        elif isinstance(data, list) and len(data) == 4:
            pts = np.array(data, dtype=np.float32)
        else:
            raise ValueError("JSON doesn't contain recognizable corners format.")

        if "input" in data and isinstance(data["input"], str):
            img_from_json = data["input"]
    else:
        raise ValueError("JSON root must be an object or a list of 4 [x,y] pairs.")

    # Ensure correct order
    pts = order_corners(pts)
    return pts, img_from_json


def maze_warp_from_json():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json", default="part_2_maze_solution/camera_capture_corners.json",
        help="Path to corners JSON"
    )
    ap.add_argument(
        "--image",
        help="Optional override for image path; defaults to 'input' inside JSON, relative to JSON file dir"
    )
    ap.add_argument(
        "--out", default="part_2_maze_solution/maze_warp.png",
        help="Output warped image path"
    )
    ap.add_argument(
        "--target-size",
        help="Force output size as WIDTHxHEIGHT (e.g., 1200x800). If omitted, inferred from quad"
    )
    ap.add_argument(
        "--pad", type=int, default=0,
        help="Uniform pixel padding around the warped output"
    )
    args = ap.parse_args()

    # Read corners and image path from JSON
    try:
        corners, img_from_json = read_corners_from_json(args.json)
    except Exception as e:
        print(f"Error reading JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # Resolve image path
    if args.image:
        img_path = args.image
    else:
        if not img_from_json:
            print(
                "No image path provided via --image and none found in JSON 'input' field.",
                file=sys.stderr
            )
            sys.exit(2)
        # If JSON includes a relative image name, resolve relative to JSON file directory
        json_dir = os.path.dirname(os.path.abspath(args.json))
        img_path = os.path.join(json_dir, img_from_json)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not read image: {img_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output size
    if args.target_size:
        try:
            W, H = parse_target_size(args.target_size)
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
    else:
        W, H = infer_size_from_quad(corners)

    warped = warp_perspective(img, corners, (W, H), pad=args.pad)
    ok = cv2.imwrite(args.out, warped)
    if not ok:
        print("Warning: failed to save warped output.", file=sys.stderr)

    print(f"Warp complete from: {img_path}")
    print(f"Size: {W}x{H}px  Pad: {args.pad}px")
    print(f"Saved to: {args.out}")


# ============================================================================
# CIRCLE COLOR DETECTION
# ============================================================================

def detect_color(frame, center, radius):
    """Detect dominant color (red or green) inside a circle."""
    # Robust 80% inner-disk mask to avoid edge noise
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask, (int(center[0]), int(center[1])),
        int(max(1.0, radius * 0.8)), 255, -1
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Mask within circle
    H = H[mask == 255]
    S = S[mask == 255]
    V = V[mask == 255]

    # Filter only valid pixels (saturated + bright enough)
    valid = (S > 50) & (V > 50)
    H = H[valid]

    if len(H) == 0:
        return "green"

    # Count pixels in red and green hue ranges
    red_mask = ((H <= 10) | (H >= 170))
    green_mask = ((H >= 35) & (H <= 85))

    red_ratio = np.sum(red_mask) / len(H) if len(H) > 0 else 0.0
    green_ratio = np.sum(green_mask) / len(H) if len(H) > 0 else 0.0

    if red_ratio > green_ratio and red_ratio > 0.1:
        return "red"
    elif green_ratio > red_ratio and green_ratio > 0.1:
        return "green"
    else:
        return "unknown"


def detect_circles_and_overlay(img_bgr, out_path):
    frame = img_bgr.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- VERY permissive Green range ---
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # --- VERY permissive Red range ---
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
        hsv, lower_red2, upper_red2
    )

    # Apply morphological operations to clean up masks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    results = []

    for mask, color_name in [(mask_green, "green"), (mask_red, "red")]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        print(f"  → Found {len(contours)} {color_name} contours")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            print(f"    - {color_name} contour area: {area:.1f}")

            if area < 100:
                continue

            # Compute centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            u = int(M["m10"] / M["m00"])
            v = int(M["m01"] / M["m00"])

            # Create a fake radius
            r = int(np.sqrt(area / np.pi))

            # Draw annotations
            cv2.circle(frame, (u, v), r, (0, 255, 0), 2)
            cv2.circle(frame, (u, v), 2, (0, 255, 0), 3)
            cv2.putText(
                frame, color_name, (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                2, cv2.LINE_AA
            )

            results.append({
                "center": [float(u), float(v)],
                "radius": float(r),
                "color": color_name
            })
            print(f"    ✓ Accepted {color_name} circle at ({u}, {v}) with area {area:.1f}")

    cv2.imwrite(out_path, frame)

    # Save debug masks to see what's being detected
    cv2.imwrite("part_2_maze_solution/debug_mask_green.png", mask_green)
    cv2.imwrite("part_2_maze_solution/debug_mask_red.png", mask_red)

    print(f"  → TOTAL Detected {len(results)} circles: {[r['color'] for r in results]}")
    return results


# ============================================================================
# GRID + WALLS
# ============================================================================

def binarize_walls(gray: np.ndarray, adaptive: bool) -> np.ndarray:
    """
    Return mask_walls (uint8) where 255 = wall (black in original), 0 = not wall.
    Otsu/Adaptive threshold with inversion.
    """
    if adaptive:
        block = 35 if min(gray.shape[:2]) > 400 else 21
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block, 5
        )
    else:
        _, th = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    return th


def morph(mask: np.ndarray, k_open: int, k_close: int) -> np.ndarray:
    m = mask.copy()
    if k_open > 0:
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_open+1, 2*k_open+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=1)
    if k_close > 0:
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (2*k_close+1, 2*k_close+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=1)
    return m


def draw_grid_lines(img: np.ndarray, grid: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    color = (255, 255, 255)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w-1, y), color, 1)
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h-1), color, 1)
    return out


def draw_grid_with_values(img: np.ndarray, grid: int, values_mat: np.ndarray,
                          font_scale: float, thickness: int) -> np.ndarray:
    out = draw_grid_lines(img, grid)
    h, w = out.shape[:2]
    gh, gw = values_mat.shape
    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            text = str(int(values_mat[gy, gx]))
            # black outline
            cv2.putText(
                out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness+2, cv2.LINE_AA
            )
            # white text
            cv2.putText(
                out, text, (cx-6, cy+5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
    return out


def maze_circles_and_grid():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", default="part_2_maze_solution/maze_warp.png",
        help="Path to maze image (white path, black wall)"
    )
    ap.add_argument("--grid", type=int, default=30, help="Cell size in pixels")
    ap.add_argument(
        "--adaptive", type=int, default=0,
        help="Use adaptive threshold (1) or Otsu (0, default)"
    )
    ap.add_argument(
        "--blur", type=int, default=5,
        help="Median blur kernel size (0=off)"
    )
    ap.add_argument(
        "--open", type=int, default=0,
        help="Morph open radius (px) to remove specks"
    )
    ap.add_argument(
        "--close", type=int, default=0,
        help="Morph close radius (px) to bridge gaps"
    )
    ap.add_argument(
        "--threshold", type=float, default=5.0,
        help="Percent of wall pixels to mark a cell as blocked (0..100)"
    )
    ap.add_argument(
        "--circles-overlay-out", default="part_2_maze_solution/circles_overlay.png",
        help="Detected circles + color labels"
    )
    ap.add_argument(
        "--grid-overlay-out", default="part_2_maze_solution/grid_overlay.png",
        help="Grid overlay (lines only)"
    )
    ap.add_argument(
        "--grid-overlay-annot-out", default="part_2_maze_solution/grid_overlay_annot.png",
        help="Grid overlay with 0/1 annotations"
    )
    ap.add_argument(
        "--walls-mask-out", default="part_2_maze_solution/walls_mask.png",
        help="Binary WALL mask (255=wall)"
    )
    ap.add_argument(
        "--json-out", default="part_2_maze_solution/result.json",
        help="Unified JSON with circles + per-cell data"
    )
    ap.add_argument(
        "--font-scale", type=float, default=0.4,
        help="Font scale for numbers"
    )
    ap.add_argument(
        "--thickness", type=int, default=1,
        help="Font thickness for numbers"
    )
    args = ap.parse_args()

    # --- Load image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    grid = max(1, args.grid)

    # 1) Detect circles + colors; save overlay
    circles_info = detect_circles_and_overlay(img, args.circles_overlay_out)

    # 2) GRID overlays + walls
    # 2a) Save grid overlay (lines only)
    grid_overlay = draw_grid_lines(img, grid)
    cv2.imwrite(args.grid_overlay_out, grid_overlay)

    # 2b) Walls mask (255=wall)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.blur > 0:
        k = max(1, args.blur | 1)
        gray = cv2.medianBlur(gray, k)
    walls_mask = binarize_walls(gray, adaptive=bool(args.adaptive))
    walls_mask = morph(walls_mask, k_open=max(0, args.open), k_close=max(0, args.close))
    cv2.imwrite(args.walls_mask_out, walls_mask)

    # 2c) Per-cell values & centers
    h, w = walls_mask.shape[:2]
    gh = (h + grid - 1) // grid
    gw = (w + grid - 1) // grid
    values_mat = np.zeros((gh, gw), dtype=np.uint8)
    cells = []
    thresh_pct = float(args.threshold)

    for gy in range(gh):
        y0 = gy * grid
        y1 = min((gy + 1) * grid, h)
        cy = int((y0 + y1) / 2)
        for gx in range(gw):
            x0 = gx * grid
            x1 = min((gx + 1) * grid, w)
            cx = int((x0 + x1) / 2)
            blk = walls_mask[y0:y1, x0:x1]
            wall_pct = (blk > 0).mean() * 100.0 if blk.size > 0 else 0.0
            value = 0 if wall_pct >= thresh_pct else 1
            values_mat[gy, gx] = value
            cells.append({
                "row": int(gy),
                "col": int(gx),
                "value": int(value),
                "center_px": [int(cx), int(cy)]
            })

    grid_w_px = gw * grid
    grid_h_px = gh * grid

    for circle in circles_info:
        cx, cy = circle["center"]
        r = circle["radius"]

        # Calculate bounding box of the circle in pixel coordinates
        x_min_px = int(cx - r/1.5)
        x_max_px = int(cx + r/1.5)
        y_min_px = int(cy - r/1.5)
        y_max_px = int(cy + r/1.5)

        # Convert bounding box coordinates to grid indices
        gx_min = max(0, min(gw - 1, x_min_px // grid))
        gx_max = max(0, min(gw - 1, (x_max_px - 1) // grid))
        gy_min = max(0, min(gh - 1, y_min_px // grid))
        gy_max = max(0, min(gh - 1, (y_max_px - 1) // grid))

        # Iterate over all grid cells that overlap with the circle's bounding box
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                values_mat[gy, gx] = 1
                cell_index = gy * gw + gx
                if cell_index < len(cells):
                    cells[cell_index]["value"] = 1

    # 2d) Annotated overlay (grid + 0/1 at centers)
    grid_overlay_annot = draw_grid_with_values(
        img, grid, values_mat, args.font_scale, args.thickness
    )
    cv2.imwrite(args.grid_overlay_annot_out, grid_overlay_annot)

    # 3) JSON export
    meta = {
        "input": args.input,
        "circles_overlay_path": args.circles_overlay_out,
        "grid_size_px": grid,
        "grid_rows": int(gh),
        "grid_cols": int(gw),
        "threshold_percent": thresh_pct,
        "grid_overlay_path": args.grid_overlay_out,
        "grid_overlay_annot_path": args.grid_overlay_annot_out,
        "walls_mask_path": args.walls_mask_out,
        "circles": circles_info,
        "cells": cells
    }
    with open(args.json_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Circles overlay saved to: {args.circles_overlay_out} ({len(circles_info)} circles)")
    print(f"Grid overlay saved to: {args.grid_overlay_out}")
    print(f"Walls mask saved to: {args.walls_mask_out}")
    print(f"Annotated grid overlay saved to: {args.grid_overlay_annot_out}")
    print(f"JSON saved to: {args.json_out}")
    print("Cell value legend: 1 = no wall, 0 = wall present")


# ============================================================================
# MAZE SOLVER
# ============================================================================

@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    value: int
    center_px: Tuple[int, int]


def load_maze(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def cells_to_grid(cells_json: List[Dict[str, Any]], rows: int, cols: int) -> List[List[Cell]]:
    grid: List[List[Cell]] = [[None for _ in range(cols)] for _ in range(rows)]
    for c in cells_json:
        cell = Cell(
            row=c["row"],
            col=c["col"],
            value=int(c["value"]),
            center_px=(int(c["center_px"][0]), int(c["center_px"][1])),
        )
        grid[cell.row][cell.col] = cell

    # quick sanity
    for r in range(rows):
        for q in range(cols):
            if grid[r][q] is None:
                raise ValueError(f"Missing cell at ({r},{q}) in JSON.")
    return grid


def nearest_cell_by_pixel(grid: List[List[Cell]], x: float, y: float) -> Tuple[int, int]:
    """Return (row, col) of the cell whose center is nearest to (x,y)."""
    best = None
    best_d2 = 1e18
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cx, cy = grid[r][c].center_px
            d2 = (cx - x) ** 2 + (cy - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (r, c)
    return best


def parse_start_end(grid, data, start_arg, end_arg):
    circles = data.get("circles", [])
    green = next((c for c in circles if c.get("color") == "green"), None)
    red = next((c for c in circles if c.get("color") == "red"), None)
    if green is None or red is None:
        raise ValueError("JSON must include green and red circles.")

    green_px = (float(green["center"][0]), float(green["center"][1]))
    red_px = (float(red["center"][0]), float(red["center"][1]))

    # Defaults: cells nearest to the green/red dots
    default_start = nearest_cell_by_pixel(grid, *green_px)
    default_end = nearest_cell_by_pixel(grid, *red_px)

    def parse_point(arg, default_rc):
        if not arg:
            return default_rc
        a = arg.strip().lower()
        if a == "green":
            return default_start
        if a == "red":
            return default_end
        if "," in a:
            r_str, c_str = a.split(",", 1)
            return (int(r_str), int(c_str))
        raise ValueError(f"Invalid start/end value: {arg}. Use 'green', 'red', or 'r,c'.")

    start_rc = parse_point(start_arg, default_start)
    end_rc = parse_point(end_arg, default_end)

    # Choose which circle pixel to use as the start/end anchor
    def choose_anchor(which, rc):
        if which and which.strip().lower() in ("green", "red"):
            return green_px if which.strip().lower() == "green" else red_px
        cx, cy = grid[rc[0]][rc[1]].center_px
        dg = (cx - green_px[0])**2 + (cy - green_px[1])**2
        dr = (cx - red_px[0])**2 + (cy - red_px[1])**2
        return green_px if dg <= dr else red_px

    start_dot = choose_anchor(start_arg, start_rc)
    end_dot = choose_anchor(end_arg, end_rc)

    return start_rc, end_rc, (int(start_dot[0]), int(start_dot[1])), (int(end_dot[0]), int(end_dot[1]))


def bfs_path(
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    BFS shortest path, 4-connected.
    Rule:
      - You can stand on start even if value=0.
      - You can stand on end even if value=0.
      - All intermediate steps must be on cells with value=1.
    """
    rows, cols = len(grid), len(grid[0])
    sr, sc = start_rc
    tr, tc = end_rc

    def neighbors(r: int, c: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                yield rr, cc

    q = deque()
    q.append((sr, sc))
    prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
    seen = set([(sr, sc)])

    while q:
        r, c = q.popleft()
        if (r, c) == (tr, tc):
            # reconstruct
            path = [(r, c)]
            while (r, c) != (sr, sc):
                r, c = prev[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        for rr, cc in neighbors(r, c):
            if (rr, cc) in seen:
                continue

            # allow stepping onto end regardless of value
            if (rr, cc) == (tr, tc):
                prev[(rr, cc)] = (r, c)
                seen.add((rr, cc))
                q.append((rr, cc))
                continue

            # otherwise, must be a 1-cell to traverse
            if grid[rr][cc].value != 1:
                continue

            prev[(rr, cc)] = (r, c)
            seen.add((rr, cc))
            q.append((rr, cc))

    return None


def draw_path_on_image(
    image_path: str,
    out_image_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
    line_width: int = 5,
) -> None:
    """
    Draws the polyline from the green dot to the first cell center,
    through the path cells, and finally to the red dot.
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    # Build pixel polyline
    poly: List[Tuple[int, int]] = []

    # Start at the green circle center
    poly.append(start_circle_px)

    # Then the centers of each cell on the path (in order)
    for (r, c) in cell_path:
        poly.append(grid[r][c].center_px)

    # Finish at the red circle center
    poly.append(end_circle_px)

    # Draw the path
    draw.line(poly, width=line_width, fill=(255, 0, 0, 255))
    
    # Mark endpoints for clarity
    r_rad = max(6, line_width * 2)
    g_rad = max(6, line_width * 2)
    gx, gy = start_circle_px
    rx, ry = end_circle_px
    draw.ellipse(
        (gx - g_rad, gy - g_rad, gx + g_rad, gy + g_rad),
        outline=(0, 255, 0, 255), width=3
    )
    draw.ellipse(
        (rx - r_rad, ry - r_rad, rx + r_rad, ry + r_rad),
        outline=(255, 0, 0, 255), width=3
    )

    img.save(out_image_path)


def write_path_json(
    out_json_path: str,
    cell_path: List[Tuple[int, int]],
    grid: List[List[Cell]],
    start_rc: Tuple[int, int],
    end_rc: Tuple[int, int],
    start_circle_px: Tuple[int, int],
    end_circle_px: Tuple[int, int],
) -> None:
    """
    Writes a JSON containing:
      - start/end pixel coordinates (green/red dot centers)
      - start/end cells
      - path as cells (row,col,value,center_px)
      - path as pixels (list of [x,y] including the dots at beginning/end)
      - length (number of moves)
    """
    # path cells expanded
    path_cells_expanded = [
        {
            "row": r,
            "col": c,
            "value": grid[r][c].value,
            "center_px": [grid[r][c].center_px[0], grid[r][c].center_px[1]],
        }
        for (r, c) in cell_path
    ]

    # pixel polyline that matches drawn line
    pixel_polyline: List[List[int]] = []
    pixel_polyline.append([start_circle_px[0], start_circle_px[1]])
    for (r, c) in cell_path:
        x, y = grid[r][c].center_px
        pixel_polyline.append([x, y])
    pixel_polyline.append([end_circle_px[0], end_circle_px[1]])

    out = {
        "start_cell": {"row": start_rc[0], "col": start_rc[1]},
        "end_cell": {"row": end_rc[0], "col": end_rc[1]},
        "start_circle_px": [start_circle_px[0], start_circle_px[1]],
        "end_circle_px": [end_circle_px[0], end_circle_px[1]],
        "path_cells": path_cells_expanded,
        "path_pixels": pixel_polyline,
        "moves": max(0, len(cell_path) - 1),
        "notes": "Path respects rule: 1-only traversal; start/end cells may be 0.",
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def solve_maze():
    ap = argparse.ArgumentParser(
        description="Solve a grid maze from JSON and draw the path."
    )
    ap.add_argument(
        "--json_path", default="part_2_maze_solution/result.json",
        help="Input maze JSON path."
    )
    ap.add_argument(
        "--start", help="Start: 'green' (default), 'red', or 'row,col'", default='red'
    )
    ap.add_argument(
        "--end", help="End: 'red' (default), 'green', or 'row,col'", default='green'
    )
    ap.add_argument(
        "--out-image", help="Output image with path (PNG).",
        default="part_2_maze_solution/solution_overlay.png"
    )
    ap.add_argument(
        "--out-json", help="Output JSON with path pixel points.",
        default="part_2_maze_solution/solution_path_points.json"
    )
    ap.add_argument(
        "--line-width", type=int, default=5,
        help="Path line width on image."
    )
    args = ap.parse_args()

    data = load_maze(args.json_path)
    rows = int(data["grid_rows"])
    cols = int(data["grid_cols"])
    grid = cells_to_grid(data["cells"], rows, cols)

    start_rc, end_rc, start_circle_px, end_circle_px = parse_start_end(
        grid, data, args.start, args.end
    )

    path = bfs_path(grid, start_rc, end_rc)
    if path is None:
        raise SystemExit("No feasible path found under the given rules.")

    # Draw overlay on the image
    image_path = data["input"]
    if not Path(image_path).exists():
        raise SystemExit(f"Input image not found at: {image_path}")

    draw_path_on_image(
        image_path=image_path,
        out_image_path=args.out_image,
        cell_path=path,
        grid=grid,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
        line_width=args.line_width,
    )

    # Write path JSON
    write_path_json(
        out_json_path=args.out_json,
        cell_path=path,
        grid=grid,
        start_rc=start_rc,
        end_rc=end_rc,
        start_circle_px=start_circle_px,
        end_circle_px=end_circle_px,
    )

    print(f"Done.\n - Path image: {args.out_image}\n - Path JSON: {args.out_json}")


# ============================================================================
# UNWARP AND OVERLAY PATH
# ============================================================================

def read_corners_json(path: str) -> Tuple[np.ndarray, str]:
    """
    Returns: (corners TL,TR,BR,BL as float32 4x2), original_image_path
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "corners" in data:
            tl = data["corners"]["TL"]
            tr = data["corners"]["TR"]
            br = data["corners"]["BR"]
            bl = data["corners"]["BL"]
            corners = np.array([tl, tr, br, bl], dtype=np.float32)
        elif all(k in data for k in ("TL", "TR", "BR", "BL")):
            corners = np.array(
                [data["TL"], data["TR"], data["BR"], data["BL"]], dtype=np.float32
            )
        else:
            raise ValueError(
                "Corners JSON must contain 'corners':{TL,TR,BR,BL} or root-level TL/TR/BR/BL."
            )

        img_path = data.get("input", "")
        if not isinstance(img_path, str) or not img_path:
            raise ValueError("Corners JSON must include original image path in 'input'.")
    else:
        raise ValueError("Invalid corners JSON root object.")

    return order_corners(corners), img_path


def read_path_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_polyline_pixels(path_data: Dict[str, Any], mode: str = "auto") -> List[List[float]]:
    """
    Returns a polyline as [[x,y], ...] in WARPED coordinates.
    Priority:
      - if mode=='auto': use 'path_pixels' if present; otherwise build from 'path_cells'
      - if mode=='pixels': use 'path_pixels'
      - if mode=='cells': build [start_circle_px] + each cell.center_px + [end_circle_px]
    """
    def has_pixels():
        return isinstance(path_data.get("path_pixels"), list) and len(path_data["path_pixels"]) > 0

    def has_cells():
        return isinstance(path_data.get("path_cells"), list) and len(path_data["path_cells"]) > 0

    if mode == "pixels" or (mode == "auto" and has_pixels()):
        return [[float(x), float(y)] for x, y in path_data["path_pixels"]]

    if mode == "cells" or (mode == "auto" and not has_pixels() and has_cells()):
        poly: List[List[float]] = []
        if "start_circle_px" in path_data:
            sx, sy = path_data["start_circle_px"]
            poly.append([float(sx), float(sy)])
        for c in path_data["path_cells"]:
            cx, cy = c["center_px"]
            poly.append([float(cx), float(cy)])
        if "end_circle_px" in path_data:
            ex, ey = path_data["end_circle_px"]
            poly.append([float(ex), float(ey)])

        if not poly:
            raise ValueError(
                "No points found in 'path_cells' and no start/end circle pixels provided."
            )
        return poly

    raise ValueError("Path JSON must contain either 'path_pixels' or 'path_cells'.")


def unwarped_and_overlay_path():
    ap = argparse.ArgumentParser(
        description="Unwarp path points and overlay them on the original image."
    )
    ap.add_argument(
        "--corners_json", default="part_2_maze_solution/camera_capture_corners.json",
        help="Corners JSON used for warping."
    )
    ap.add_argument(
        "--path_json", default="part_2_maze_solution/solution_path_points.json",
        help="Path JSON produced by solver."
    )
    ap.add_argument(
        "--warped-image", default="part_2_maze_solution/maze_warp.png",
        help="Path to the warped image used when solving."
    )
    ap.add_argument(
        "--from", dest="from_mode", default="auto", choices=["auto", "pixels", "cells"],
        help="Whether to read 'path_pixels' or rebuild from 'path_cells'."
    )
    ap.add_argument(
        "--out-image", default="part_2_maze_solution/original_with_path.png",
        help="Output overlay on the ORIGINAL image."
    )
    ap.add_argument(
        "--out-json", default="part_2_maze_solution/solution_path_points_unwarped.json",
        help="Output JSON with unwarped points."
    )
    ap.add_argument(
        "--line-width", type=int, default=5,
        help="Overlay path width."
    )
    args = ap.parse_args()

    # Load corners + original image path
    corners, orig_img_path = read_corners_json(args.corners_json)

    # Resolve original image path
    if not os.path.isabs(orig_img_path):
        cj_dir = os.path.dirname(os.path.abspath(args.corners_json))
        orig_img_path = os.path.join(cj_dir, orig_img_path)

    orig_img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if orig_img is None:
        raise SystemExit(f"Could not read original image: {orig_img_path}")

    # Determine the warped image size
    warped_image_path = args.warped_image or "maze_warp.png"
    warped_img = cv2.imread(warped_image_path, cv2.IMREAD_COLOR)
    if warped_img is None:
        raise SystemExit(f"Could not read warped image '{warped_image_path}'.")
    H, W = warped_img.shape[:2]

    # Build inverse perspective transform
    src_rect = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    dst_quad = corners.astype(np.float32)
    M_inv = cv2.getPerspectiveTransform(src_rect, dst_quad)

    # Load path JSON and collect a warped-space polyline
    path_data = read_path_json(args.path_json)
    poly_warped = np.array(
        collect_polyline_pixels(path_data, mode=args.from_mode),
        dtype=np.float32
    ).reshape(-1, 1, 2)

    # Transform to original-image coordinates
    poly_orig = cv2.perspectiveTransform(poly_warped, M_inv).reshape(-1, 2)

    # Draw the unwarped path on the original image
    img_overlay = orig_img.copy()
    poly_orig_int = [(int(round(x)), int(round(y))) for x, y in poly_orig]

    # Draw the polyline
    for i in range(len(poly_orig_int) - 1):
        cv2.line(
            img_overlay, poly_orig_int[i], poly_orig_int[i + 1],
            color=(0, 0, 255), thickness=args.line_width
        )
    
    # Mark endpoints
    if len(poly_orig_int) >= 1:
        cv2.circle(
            img_overlay, poly_orig_int[0], max(6, args.line_width * 2),
            (0, 255, 0), thickness=3
        )
    if len(poly_orig_int) >= 2:
        cv2.circle(
            img_overlay, poly_orig_int[-1], max(6, args.line_width * 2),
            (0, 0, 255), thickness=3
        )

    ok = cv2.imwrite(args.out_image, img_overlay)
    if not ok:
        print("Warning: failed to save overlay image.", flush=True)

    # Save unwarped points JSON
    out = {
        "source_corners_json": os.path.abspath(args.corners_json),
        "source_path_json": os.path.abspath(args.path_json),
        "original_image": os.path.abspath(orig_img_path),
        "warped_image": os.path.abspath(warped_image_path),
        "unwarped_path_pixels": [[int(x), int(y)] for (x, y) in poly_orig_int],
        "notes": "Polyline points are in ORIGINAL image pixel coordinates."
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Done.")
    print(f" - Overlay image (original space): {args.out_image}")
    print(f" - Unwarped path JSON: {args.out_json}")


# ============================================================================
# AFFINE TRANSFORMATION AND ROBOT MOTION
# ============================================================================

def fit_affine(img_pts, rob_xy):
    M, inliers = cv2.estimateAffine2D(
        img_pts.reshape(-1, 1, 2),
        rob_xy.reshape(-1, 1, 2),
        ransacReprojThreshold=1.0,
        refineIters=1000
    )
    if M is None:
        raise RuntimeError("Affine estimation failed. Points may be degenerate.")
    return M


def fit_homography(img_pts, rob_xy):
    H, inliers = cv2.findHomography(
        img_pts, rob_xy, method=cv2.RANSAC, ransacReprojThreshold=1.0
    )
    if H is None:
        raise RuntimeError("Homography estimation failed. Points may be degenerate.")
    return H


def apply_affine(M, u, v):
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    XY = M @ uv1
    return float(XY[0]), float(XY[1])


def apply_homography(H, u, v):
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    Xp, Yp, W = H @ uv1
    if abs(W) < 1e-12:
        raise ZeroDivisionError("Homography scale ~ 0 for this point.")
    return float(Xp / W), float(Yp / W)


def rms_error_affine(M, img_pts, rob_xy):
    ones = np.ones((img_pts.shape[0], 1))
    uv1 = np.hstack([img_pts, ones])
    pred = (uv1 @ M.T)
    err = rob_xy - pred
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def rms_error_homography(H, img_pts, rob_xy):
    uv1 = np.hstack([img_pts, np.ones((img_pts.shape[0], 1))])
    proj = (uv1 @ H.T)
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = rob_xy - proj_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


# ============================================================================
# ROBOT MOTION UTILITIES
# ============================================================================

def move_to_home(device):
    print("Homing the robot...")
    device.move_to(240, 0, 150, 0)
    time.sleep(2)
    try:
        (pose, joint) = device.pose()
        print(f"pose: {pose}, j: {joint}")
    except:
        print("Robot at home position")


def move_to_specific_position(device, x, y, z, r=0.0):
    device.speed(50, 50)
    device.move_to(x, y, z, r)
    time.sleep(2)


def get_current_pose(device):
    time.sleep(1)
    print("current pose")
    try:
        (pose, joint) = device.pose()
        return pose, joint
    except Exception as e:
        print(f"Could not get pose: {e}")
        return None, None


def move_robot_point(device, M, u, v):
    """Move robot to pixel coordinate (u,v) using affine transformation."""
    Xa, Ya = apply_affine(M, u, v)
    print(f"Affine:  pixel({u:.3f}, {v:.3f}) -> robot({Xa:.6f}, {Ya:.6f})")
    move_to_specific_position(device, x=Xa, y=Ya, z=-45)
    time.sleep(1)


# ============================================================================
# CALIBRATED TRANSFORMATION MATRICES
# ============================================================================

M = np.array([
    [3.01998979e-02, -5.10669426e-01, 4.10088515e+02],
    [-4.61375311e-01, -9.61788211e-03, 1.07789293e+02]
], dtype=np.float64)

H = np.array([
    [-2.44594058e-02, -4.75669460e-01, 3.67247188e+02],
    [-4.34041615e-01, 5.08065338e-03, 1.20901686e+02],
    [-5.98330506e-05, -7.62411614e-05, 1.00000000e+00]
], dtype=np.float64)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # STEP 0: Home the robot FIRST
    print("=" * 70)
    print("STEP 0: HOMING ROBOT")
    print("=" * 70)
    move_to_home(device)
    time.sleep(2)

    # STEP 1-5: Image capture and processing
    print("\n" + "=" * 70)
    print("STEP 1-5: IMAGE CAPTURE AND MAZE SOLVING")
    print("=" * 70)
    capture_image_main()
    maze_warp_from_json()
    maze_circles_and_grid()
    solve_maze()
    unwarped_and_overlay_path()

    # STEP 6: Robot motion through the maze
    print("\n" + "=" * 70)
    print("STEP 6: ROBOT MAZE TRAVERSAL")
    print("=" * 70)
    device.speed(50, 50)
    move_to_home(device)
    time.sleep(2)

    # Initialize an empty list for coordinates
    pixel_coords = []

    # Load the path coordinates
    json_path = Path("part_2_maze_solution/solution_path_points_unwarped.json")

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'unwarped_path_pixels' in data:
                pixel_coords = data['unwarped_path_pixels']
            elif isinstance(data, list):
                pixel_coords = data
            else:
                print(
                    f"Error: JSON file '{json_path}' loaded, but expected list or "
                    f"object with 'unwarped_path_pixels' key."
                )

        print(f"✓ Successfully loaded {len(pixel_coords)} waypoints from {json_path}")
        print(f"\nStarting maze traversal...")

    except FileNotFoundError:
        print(f"✗ Error: The file '{json_path}' was not found.")
        device.close()
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"✗ Error: Could not decode JSON from '{json_path}'.")
        device.close()
        sys.exit(1)
    except Exception as e:
        print(f"✗ An unexpected error occurred: {e}")
        device.close()
        sys.exit(1)

    # Execute the path
    for i, (u, v) in enumerate(pixel_coords):
        print(f"[{i+1}/{len(pixel_coords)}] ", end="")
        move_robot_point(device, M, u, v)

    print("\n" + "=" * 70)
    print("✓ MAZE SOLVED! Robot completed the path.")
    print("=" * 70)

    device.close()
    print("Robot connection closed.")
