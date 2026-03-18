import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import csv
from datetime import datetime
import os
import sys


import ultralytics.utils.patches as _ulp
import cv2 as _cv2_real
_ulp.imshow = _cv2_real.imshow

MODEL_PATH         = 'weights/best.pt'
CONF_THRESH        = 0.40
ZONE_MULTIPLIER    = 1.8
UNATTENDED_TIMEOUT = 30
LOG_FILE           = 'alert_log.csv'
OUTPUT_VIDEO       = 'output_annotated.mp4'

# {0: 'Forklift', 1: 'Gloves', 2: 'Hard_hat',
#  3: 'Mask',     4: 'Person', 5: 'Safety_boots', 6: 'Vest'}
PERSON_CLASSES   = [4]
PPE_CLASSES      = [1, 2, 3, 5, 6]
FORKLIFT_CLASSES = [0]

CLASS_NAMES = {
    0: 'Forklift',
    1: 'Gloves',
    2: 'Hard_hat',
    3: 'Mask',
    4: 'Person',
    5: 'Safety_boots',
    6: 'Vest'
}

ALERT_COLORS = {
    'COMPLIANT':           (0, 255, 0),
    'PPE_VIOLATION_SAFE':  (0, 255, 255),
    'AUTHORIZED_OPERATOR': (255, 200, 0),
    'CRITICAL_VIOLATION':  (0, 0, 255),
    'UNATTENDED_MACHINE':  (0, 140, 255),
}

ALERT_PRIORITY = {
    'CRITICAL_VIOLATION':  3,
    'UNATTENDED_MACHINE':  2,
    'PPE_VIOLATION_SAFE':  1,
    'AUTHORIZED_OPERATOR': 0,
    'COMPLIANT':           0,
}

GUI_AVAILABLE = None

def check_gui():
    global GUI_AVAILABLE
    if GUI_AVAILABLE is not None:
        return GUI_AVAILABLE
    try:
        test = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('__test__', test)
        cv2.waitKey(1)
        cv2.destroyWindow('__test__')
        GUI_AVAILABLE = True
    except Exception:
        GUI_AVAILABLE = False
        print("  [INFO] No display — saving output to file only.")
    return GUI_AVAILABLE


def get_danger_zone(box, multiplier=ZONE_MULTIPLIER):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = (x2 - x1) * multiplier
    h = (y2 - y1) * multiplier
    return (int(cx - w / 2), int(cy - h / 2),
            int(cx + w / 2), int(cy + h / 2))


def is_inside_zone(person_box, zone):
    px = (person_box[0] + person_box[2]) / 2
    py = (person_box[1] + person_box[3]) / 2
    return zone[0] < px < zone[2] and zone[1] < py < zone[3]


def person_has_ppe(person_box, ppe_detections):
    px1, py1, px2, py2 = person_box
    found = []
    for ppe in ppe_detections:
        bx1, by1, bx2, by2 = ppe['box']
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        if px1 < cx < px2 and py1 < cy < py2:
            found.append(ppe['name'])
    return len(found) > 0, found

def draw_box(frame, box, label, color, conf=None):
    fh, fw = frame.shape[:2]
    x1 = max(0, box[0])
    y1 = max(0, box[1])
    x2 = min(fw - 1, box[2])
    y2 = min(fh - 1, box[3])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {conf:.2f}" if conf is not None else label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

    # Draw label below box if too close to top edge
    if y1 - th - 8 < 0:
        lx1, ly1 = x1, y2
        lx2, ly2 = min(fw - 1, x1 + tw + 6), min(fh - 1, y2 + th + 8)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(frame, text, (lx1 + 3, ly2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
    else:
        cv2.rectangle(frame,
                      (x1, y1 - th - 8),
                      (min(fw - 1, x1 + tw + 6), y1),
                      color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)


def draw_danger_zone(frame, zone, is_critical=False):
    fh, fw = frame.shape[:2]
    zx1 = max(0, zone[0])
    zy1 = max(0, zone[1])
    zx2 = min(fw - 1, zone[2])
    zy2 = min(fh - 1, zone[3])

    color = (0, 0, 255) if is_critical else (0, 140, 255)

    overlay = frame.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
    cv2.putText(frame, '! DANGER ZONE',
                (zx1 + 6, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def draw_hud(frame, frame_counts, frame_num, fps, total):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (315, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, 'Warehouse Safety Monitor',
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    total_str = f"/{total}" if total > 0 else ""
    cv2.putText(frame, f'Frame: {frame_num}{total_str}   FPS: {fps:.1f}',
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.putText(frame,
                f"CRITICAL:       {frame_counts.get('CRITICAL_VIOLATION', 0)}",
                (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame,
                f"PPE VIOLATIONS: {frame_counts.get('PPE_VIOLATION_SAFE', 0)}",
                (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame,
                f"UNATTENDED:     {frame_counts.get('UNATTENDED_MACHINE', 0)}",
                (8, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)


def draw_legend(frame, total_counts):
    items = [
        ('CRITICAL   - No PPE in forklift zone',  (0, 0, 255),   'CRITICAL_VIOLATION'),
        ('AUTHORIZED - PPE ok in forklift zone',  (255, 200, 0), 'AUTHORIZED_OPERATOR'),
        ('PPE VIOLATION - outside zone',          (0, 255, 255), 'PPE_VIOLATION_SAFE'),
        ('COMPLIANT  - PPE ok, outside zone',     (0, 255, 0),   'COMPLIANT'),
        ('DANGER ZONE boundary',                  (0, 140, 255), 'UNATTENDED_MACHINE'),
    ]

    fh, fw   = frame.shape[:2]
    row_h    = 24
    legend_h = len(items) * row_h + 12
    legend_w = 435

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, fh - legend_h), (legend_w, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    for i, (text, color, key) in enumerate(items):
        y     = fh - legend_h + i * row_h + 20
        count = total_counts.get(key, 0)
        cv2.rectangle(frame, (8,  y - 13), (23, y + 2), color, -1)
        cv2.rectangle(frame, (8,  y - 13), (23, y + 2), (255, 255, 255), 1)
        cv2.putText(frame, f"{text}  [{count}]", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

def process_frame(frame, model):
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]

    persons   = []
    ppe_dets  = []
    forklifts = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf   = float(box.conf)
        coords = [int(x) for x in box.xyxy[0].tolist()]
        name   = CLASS_NAMES[cls_id]

        if cls_id in PERSON_CLASSES:
            persons.append({'box': coords, 'conf': conf})
        elif cls_id in PPE_CLASSES:
            ppe_dets.append({'box': coords, 'conf': conf, 'name': name})
            draw_box(frame, coords, name, (180, 255, 180), conf)
        elif cls_id in FORKLIFT_CLASSES:
            forklifts.append({'box': coords, 'conf': conf})

    alerts       = []
    frame_counts = defaultdict(int)

    if forklifts:
        for fi, forklift in enumerate(forklifts):
            draw_box(frame, forklift['box'], 'Forklift', (200, 200, 200), forklift['conf'])

            zone              = get_danger_zone(forklift['box'])
            zone_has_critical = False
            operator_present  = False

            for person in persons:
                has_ppe, ppe_list = person_has_ppe(person['box'], ppe_dets)
                in_zone           = is_inside_zone(person['box'], zone)

                if in_zone and not has_ppe:
                    state             = 'CRITICAL_VIOLATION'
                    zone_has_critical = True
                    operator_present  = True
                elif in_zone and has_ppe:
                    state            = 'AUTHORIZED_OPERATOR'
                    operator_present = True
                elif not in_zone and not has_ppe:
                    state = 'PPE_VIOLATION_SAFE'
                else:
                    state = 'COMPLIANT'

                draw_box(frame, person['box'], state, ALERT_COLORS[state], person['conf'])
                frame_counts[state] += 1
                alerts.append({
                    'state':     state,
                    'in_zone':   in_zone,
                    'has_ppe':   has_ppe,
                    'ppe_items': ', '.join(ppe_list) if ppe_list else 'None'
                })

            draw_danger_zone(frame, zone, zone_has_critical)
            forklifts[fi]['operator_present'] = operator_present

    else:
        # No forklift — PPE only check
        for person in persons:
            has_ppe, ppe_list = person_has_ppe(person['box'], ppe_dets)
            state = 'COMPLIANT' if has_ppe else 'PPE_VIOLATION_SAFE'
            draw_box(frame, person['box'], state, ALERT_COLORS[state], person['conf'])
            frame_counts[state] += 1
            alerts.append({
                'state':     state,
                'in_zone':   False,
                'has_ppe':   has_ppe,
                'ppe_items': ', '.join(ppe_list) if ppe_list else 'None'
            })

    return frame, alerts, forklifts, frame_counts

def main():
    print("=" * 45)
    print("   Warehouse Safety Monitor — Video Mode")
    print("=" * 45)

    path = input("\n  Enter video path (e.g. test_video.mp4): ").strip()
    if not os.path.exists(path):
        print(f"\n  File not found: {path}")
        sys.exit(1)

    print(f"\n  Loading model...")
    model = YOLO(MODEL_PATH)
    print(f"  Model loaded — classes: {model.names}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"\n  Cannot open video: {path}")
        sys.exit(1)

    fps_video = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration  = total / fps_video if fps_video > 0 else 0

    print(f"\n  Video info:")
    print(f"    Resolution : {w}x{h}")
    print(f"    FPS        : {fps_video}")
    print(f"    Frames     : {total}")
    print(f"    Duration   : {duration:.1f}s")
    print(f"    Output     : {OUTPUT_VIDEO}")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_video, (w, h)
    )

    gui = check_gui()
    if gui:
        print(f"\n  Live window active — press Q to stop early.")
    else:
        print(f"\n  No GUI — writing annotated video to {OUTPUT_VIDEO}")
        print(f"  Press Ctrl+C to stop early.")

    print(f"\n  Processing...\n")

    total_counts          = defaultdict(int)
    machine_last_operator = {}
    log_rows              = []
    frame_num             = 0
    t_prev                = time.time()
    t_start               = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            t_now    = time.time()
            fps_live = 1.0 / (t_now - t_prev + 1e-6)
            t_prev   = t_now

            frame, alerts, forklifts, frame_counts = process_frame(frame, model)

            for state, cnt in frame_counts.items():
                total_counts[state] += cnt

            for a in alerts:
                log_rows.append({
                    'timestamp': datetime.now().isoformat(),
                    'frame':     frame_num,
                    **a
                })

            for fi, forklift in enumerate(forklifts):
                if not forklift.get('operator_present', False):
                    last_seen = machine_last_operator.get(fi, t_now)
                    elapsed   = t_now - last_seen
                    if elapsed > UNATTENDED_TIMEOUT:
                        frame_counts['UNATTENDED_MACHINE'] += 1
                        total_counts['UNATTENDED_MACHINE'] += 1
                        fx1, fy1, fx2, fy2 = forklift['box']
                        cv2.putText(frame,
                                    f'UNATTENDED {elapsed:.0f}s',
                                    (fx1, fy2 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (0, 140, 255), 2)
                        log_rows.append({
                            'timestamp': datetime.now().isoformat(),
                            'frame':     frame_num,
                            'state':     'UNATTENDED_MACHINE',
                            'in_zone':   False,
                            'has_ppe':   False,
                            'ppe_items': 'N/A'
                        })
                else:
                    machine_last_operator[fi] = t_now

            draw_hud(frame, frame_counts, frame_num, fps_live, total)
            draw_legend(frame, total_counts)

            writer.write(frame)

            pct = (frame_num / total * 100) if total > 0 else 0
            bar = ('=' * int(pct / 5)).ljust(20)
            print(f"\r  [{bar}] {pct:5.1f}%  Frame {frame_num}/{total}  FPS:{fps_live:.1f}  "
                  f"CRIT:{frame_counts.get('CRITICAL_VIOLATION',0)}  "
                  f"PPE_VIOL:{frame_counts.get('PPE_VIOLATION_SAFE',0)}  ",
                  end='', flush=True)

            if gui:
                cv2.imshow('Warehouse Safety Monitor', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n  Stopped early by user.")
                    break

    except KeyboardInterrupt:
        print("\n\n  Stopped early (Ctrl+C).")

    cap.release()
    writer.release()
    if gui:
        cv2.destroyAllWindows()

    elapsed_total = time.time() - t_start

    if log_rows:
        with open(LOG_FILE, 'w', newline='') as f:
            dw = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            dw.writeheader()
            dw.writerows(log_rows)

    print(f"\n\n{'=' * 45}")
    print(f"   SESSION SUMMARY")
    print(f"{'=' * 45}")
    print(f"  Frames processed : {frame_num}")
    print(f"  Time taken       : {elapsed_total:.1f}s")
    print(f"  Avg FPS          : {frame_num / elapsed_total:.1f}")
    print()
    for state, count in sorted(total_counts.items(),
                                key=lambda x: -ALERT_PRIORITY.get(x[0], 0)):
        bar   = '#' * min(count // 10, 30)
        print(f"  {state:<30} {count:>5}  {bar}")
    print()
    print(f"  Log   -> {LOG_FILE}")
    print(f"  Video -> {OUTPUT_VIDEO}")
    print(f"{'=' * 45}")

    if not gui and os.path.exists(OUTPUT_VIDEO):
        print(f"\n  Opening output video...")
        os.startfile(os.path.abspath(OUTPUT_VIDEO))


if __name__ == "__main__":
    main()
