import cv2
import os
import csv
import time
import glob
import math
import numpy as np
import torch
from datetime import datetime

# Official built-in Ultralytics SAM interface
from ultralytics import SAM


def initialize_sam3(checkpoint_path="sam3.pt"):
    """
    Initializes the official Ultralytics SAM engine.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing native Ultralytics SAM on device: {device.upper()}...")
    return SAM(checkpoint_path)


def parse_gaze_txt_file(txt_path):
    """
    Parses eye-tracking coordinates out of your specific .txt structure.
    Maps frame sequences to absolute (x, y) coordinates.
    """
    gaze_lookup = {}
    if not txt_path or not os.path.exists(txt_path):
        return None
        
    try:
        with open(txt_path, "r") as f:
            # Skip the header row
            next(f, None)
            
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  
                    
                parts = line.replace("\t", ",").replace(" ", ",").split(",")
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) >= 4:
                    try:
                        # parts[0] is frame_etg, parts[1] is frame_gar
                        frame_idx = int(float(parts[0]))
                        
                        # Fix: X and Y are at index 2 and 3
                        gaze_x = float(parts[2])
                        gaze_y = float(parts[3])
                        
                        gaze_lookup[frame_idx] = (gaze_x, gaze_y)
                    except ValueError:
                        continue
                        
        return gaze_lookup
    except Exception as e:
        print(f"  Error parsing text coordinate file {txt_path}: {e}")
        return None


def evaluate_frame_with_gaze(frame, predictor, gaze_coords):
    """
    Evaluates a frame using the human gaze coordinate as a visual point prompt.
    """
    is_hit = 0
    distance = -1.0
    
    if gaze_coords is None:
        return is_hit, distance

    gaze_x, gaze_y = gaze_coords

    if math.isnan(gaze_x) or math.isnan(gaze_y):
        return is_hit, distance

    frame_h, frame_w = frame.shape[:2]
    check_x = min(max(int(gaze_x), 0), frame_w - 1)
    check_y = min(max(int(gaze_y), 0), frame_h - 1)

    # Point prompt interaction layer inside SAM 3
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
        results = predictor(frame, points=[[check_x, check_y]], labels=[1], imgsz=644, verbose=False)[0]
    
    if hasattr(results, 'masks') and results.masks is not None and len(results.masks.data) > 0:
        mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
        mask_h, mask_w = mask.shape
        
        if (mask_h, mask_w) != (frame_h, frame_w):
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)

        # 1. Metric: Fixation-in-Mask Hit Rate
        if mask[check_y, check_x] > 0:
            is_hit = 1

        # 2. Metric: Centroid Drift
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            centroid_x = np.mean(x_indices)
            centroid_y = np.mean(y_indices)
            distance = float(np.sqrt((gaze_x - centroid_x)**2 + (gaze_y - centroid_y)**2))

    return is_hit, distance


def test_single_video(video_path, eye_tracking_lookup, predictor, sample_rate):
    """
    Iterates over video frames. Benchmarks SAM metrics every `sample_rate` frames, 
    but tracks hardware reliability (Track Loss & TTR) continuously on every frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Could not open video file {video_path}")
        return None

    # Get video properties for time calculations
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    total_frames_sampled = 0
    hit_count = 0
    valid_distances = []
    
    # New reliability trackers
    misses = 0
    current_loss_streak = 0
    recovery_times_ms = []

    while cap.isOpened():
        gaze_coords = eye_tracking_lookup.get(frame_idx, None) if eye_tracking_lookup else None
        
        # --- 1. Reliability Metrics (Calculated continuously on EVERY frame) ---
        is_valid_gaze = False
        if gaze_coords is not None:
            gx, gy = gaze_coords
            if not (math.isnan(gx) or math.isnan(gy)):
                if (0 <= gx <= width) and (0 <= gy <= height):
                    is_valid_gaze = True
                    
        if not is_valid_gaze:
            misses += 1
            current_loss_streak += 1
        else:
            if current_loss_streak > 0:
                ttr_ms = (current_loss_streak / fps) * 1000
                recovery_times_ms.append(ttr_ms)
                current_loss_streak = 0
        
        # --- 2. OpenCV Optimization & SAM Performance Metrics ---
        is_sample_frame = (frame_idx % sample_rate == 0)
        
        if is_sample_frame and is_valid_gaze:
            # We need to evaluate this frame, so fully decode it
            ret, frame = cap.read()
            if not ret:
                break
                
            is_hit, distance = evaluate_frame_with_gaze(frame, predictor, gaze_coords)
            
            total_frames_sampled += 1
            hit_count += is_hit
            if distance >= 0:
                valid_distances.append(distance)
        else:
            # We do NOT need to evaluate this frame. 
            # Grab() advances the video pointer instantly without CPU decoding overhead.
            ret = cap.grab()
            if not ret:
                break
            
        frame_idx += 1

    cap.release()
    
    # Handle edge case if video ends while the tracker is still lost
    if current_loss_streak > 0:
        ttr_ms = (current_loss_streak / fps) * 1000
        recovery_times_ms.append(ttr_ms)

    # Compile final metrics
    actual_total_frames = frame_idx 
    hit_rate_perc = (hit_count / total_frames_sampled * 100) if total_frames_sampled > 0 else 0.0
    avg_drift = np.mean(valid_distances) if valid_distances else -1.0
    
    track_loss_perc = (misses / actual_total_frames * 100) if actual_total_frames > 0 else 0.0
    max_ttr_ms = np.max(recovery_times_ms) if recovery_times_ms else 0.0

    return {
        "hit_rate_perc": round(hit_rate_perc, 2),
        "avg_drift_pixl": round(avg_drift, 2),
        "track_loss_perc": round(track_loss_perc, 2), # Control variable
        "max_ttr_ms": round(max_ttr_ms, 2) # Control
    }


def get_matching_txt_path(video_filename, txt_dir):
    """
    Fuzzy matching logic to correctly pair video conditions with text coordinates.
    Handles '01_gaussian_blur_mild.avi', '01.avi', or '1.avi' matching to '01.txt'.
    """
    base_name = os.path.splitext(video_filename)[0]
    
    # Extract only the numeric string out of the leading portion of the video name
    raw_prefix = base_name.split("_")[0]
    numeric_id = "".join(filter(str.isdigit, raw_prefix))
    
    if not numeric_id:
        return None

    # Check variation A: Direct integer match padded with a leading zero (e.g., '01.txt')
    padded_name = f"{int(numeric_id):02d}.txt"
    path_padded = os.path.join(txt_dir, padded_name)
    if os.path.exists(path_padded):
        return path_padded

    # Check variation B: Raw digit string match (e.g., '1.txt')
    path_raw = os.path.join(txt_dir, f"{int(numeric_id)}.txt")
    if os.path.exists(path_raw):
        return path_raw

    # Check variation C: Direct match fallback
    path_direct = os.path.join(txt_dir, f"{base_name}.txt")
    if os.path.exists(path_direct):
        return path_direct

    return None


def run_experiment_pipeline(base_thesis_dir, predictor, sample_rate):
    """
    Scans folders, maps videos dynamically to text baselines, and logs to /results.
    """
    drive_dir = os.path.join(base_thesis_dir, "DR(eye)VE")
    original_dir = os.path.join(drive_dir, "original")
    modified_dir = os.path.join(drive_dir, "modified")
    txt_dir = os.path.join(drive_dir, "txt")
    
    results_dir = os.path.join(base_thesis_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m-%d_%Hh%Mm")
    csv_output_path = os.path.join(results_dir, f"sam3_s{sample_rate}_{timestamp}.csv")
    
    # Collect videos
    original_videos = glob.glob(os.path.join(original_dir, "*.avi"))
    modified_videos = glob.glob(os.path.join(modified_dir, "*.avi"))
    
    all_videos = [("Original", p) for p in original_videos] + [("Modified", p) for p in modified_videos]
    
    if not all_videos:
        print(f"Error: No videos found. Check directories inside:\n {drive_dir}")
        return

    headers = ["type", "filename", "hit_rate_perc", "avg_drift_pixl", "track_loss_perc", "max_ttr_ms"]
    
    print(f"Discovered {len(all_videos)} total videos inside experimental subdirectories.")
    
    with open(csv_output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for idx, (source_type, video_path) in enumerate(sorted(all_videos, key=lambda x: os.path.basename(x[1])), 1):
            filename = os.path.basename(video_path)
            txt_path = get_matching_txt_path(filename, txt_dir)
            
            if not txt_path:
                print(f"  [{idx}/{len(all_videos)}] SKIPPED: {filename} -> Could not find matching text file in {txt_dir}")
                continue
                
            print(f"  [{idx}/{len(all_videos)}] Processing [{source_type}]: {filename} paired with {os.path.basename(txt_path)}")
            
            eye_tracking_lookup = parse_gaze_txt_file(txt_path)
            metrics = test_single_video(video_path, eye_tracking_lookup, predictor, sample_rate=sample_rate)
            
            if metrics is not None:
                writer.writerow([
                    source_type,
                    filename,
                    metrics["hit_rate_perc"],
                    metrics["avg_drift_pixl"],
                    metrics["track_loss_perc"],
                    metrics["max_ttr_ms"]
                ])
                csv_file.flush()

    print(f"\nExecution finished! Clean results compiled at:\n{csv_output_path}")


if __name__ == "__main__":
    current_thesis_dir = os.path.dirname(os.path.abspath(__file__))
    model_checkpoint = os.path.join(current_thesis_dir, "sam3.pt")

    SAMPLE_RATE = 10 # Set the sample rate for frame evaluation
    
    sam3_predictor = initialize_sam3(checkpoint_path=model_checkpoint)
    run_experiment_pipeline(current_thesis_dir, sam3_predictor, sample_rate=SAMPLE_RATE)