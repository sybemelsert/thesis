import cv2
import os
import glob
import numpy as np


# CORE FUNCTIONS FOR VIDEO ADJUSMENT

def apply_gaussian_blur(frame, severity):
    if severity == 0: return frame
    kernel_size = severity if severity % 2 != 0 else severity + 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def apply_motion_blur(frame, severity):
    if severity == 0: return frame
    kernel_size = max(1, severity)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(frame, -1, kernel)

def apply_overexposure(frame, severity):
    frame_float = frame.astype(np.float32) + severity
    return np.clip(frame_float, 0, 255).astype(np.uint8)

def apply_underexposure(frame, severity):
    frame_float = frame.astype(np.float32) - severity
    return np.clip(frame_float, 0, 255).astype(np.uint8)

def apply_occlusion(frame, gaze_x, gaze_y, severity):
    if severity == 0: return frame
    h, w, _ = frame.shape
    processed_frame = frame.copy()
    
    x1 = int(max(0, gaze_x - severity // 2))
    y1 = int(max(0, gaze_y - severity // 2))
    x2 = int(min(w, gaze_x + severity // 2))
    y2 = int(min(h, gaze_y + severity // 2))
    
    processed_frame[y1:y2, x1:x2] = [128, 128, 128]
    return processed_frame


# .txt-file parsing

def parse_gaze_txt(txt_path):
    gaze_dict = {}
    with open(txt_path, "r") as file:
        for line in file:
            line_str = line.strip()
            if not line_str:
                continue
                
            if "," in line_str:
                parts = line_str.split(",")
            else:
                parts = line_str.split()
                
            try:
                frame_idx = int(float(parts[0]))
                gaze_x = float(parts[1])
                gaze_y = float(parts[2])
                gaze_dict[frame_idx] = (gaze_x, gaze_y)
            except (ValueError, IndexError):
                continue
    return gaze_dict

def process_single_pair(video_path, txt_path, output_dir, configurations):
    """
    Process the matching .avi and .txt files inside the same folder and loop through effects
    """
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    gaze_dict = parse_gaze_txt(txt_path)
    
    if not gaze_dict:
        print(f"Warning: Gaze coordinate file {txt_path} yielded no valid data.")

    # Loop through each effect type
    for effect_name, severity_levels in configurations.items():
        # Loop through the three levels of severity
        for tier, severity_val in severity_levels.items():
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video target {video_path}")
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Form output filenames that include type and tier of the video
            output_filename = f"{base_name}_{effect_name}_{tier}.avi"
            output_path = os.path.join(output_dir, output_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"    Generating -> {output_filename} (Value: {severity_val})...")
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if effect_name == "gaussian_blur":
                    processed = apply_gaussian_blur(frame, severity_val)
                elif effect_name == "motion_blur":
                    processed = apply_motion_blur(frame, severity_val)
                elif effect_name == "overexposure":
                    processed = apply_overexposure(frame, severity_val)
                elif effect_name == "underexposure":
                    processed = apply_underexposure(frame, severity_val)
                elif effect_name == "occlusion":
                    gaze_x, gaze_y = gaze_dict.get(frame_idx, (width // 2, height // 2))
                    processed = apply_occlusion(frame, gaze_x, gaze_y, severity_val)
                else:
                    processed = frame
                    
                out_writer.write(processed)
                frame_idx += 1
                
            cap.release()
            out_writer.release()


# Automatically match .avi and .txt files within same folder

def run_batch_conversion(target_folder, experiment_configs):
    """
    Scans a directory for matching .avi and .txt file pairs and triggers processing.
    """

    # Checks for both lower and uppercase file extension variants natively
    avi_files = glob.glob(os.path.join(target_folder, "*.avi")) + glob.glob(os.path.join(target_folder, "*.AVI"))
    
    if not avi_files:
        print(f"No .avi files found in '{target_folder}'. Please check your paths.")
        return

    print(f"Found {len(avi_files)} video files. Checking for matching coordinate files...")
    
    pairs_found = 0
    for video_path in avi_files:
        base_without_ext = os.path.splitext(video_path)[0]
        expected_txt_path = base_without_ext + ".txt"
        
        if os.path.exists(expected_txt_path):
            pairs_found += 1
            print(f"\n[Pair #{pairs_found}] Processing base name: {os.path.basename(base_without_ext)}")
            
            output_folder = os.path.join(target_folder, "adversarial_outputs")
            
            process_single_pair(
                video_path=video_path,
                txt_path=expected_txt_path,
                output_dir=output_folder,
                configurations=experiment_configs
            )
        else:
            print(f"\nSkipping: Found '{os.path.basename(video_path)}' but no matching '{os.path.basename(expected_txt_path)}' exists.")

    print(f"\nConversion complete! Total matched pairs processed: {pairs_found}")


if __name__ == "__main__":
    # Automatically tracks local script directory path
    current_working_directory = os.path.dirname(os.path.abspath(__file__)) 
    
    # Experiment configurations
    experiment_configs = {
        "gaussian_blur": {
            "mild": 9,
            "medium": 25,
            "severe": 55
        },
        "motion_blur": {
            "mild": 5,
            "medium": 19,
            "severe": 35
        },
        "overexposure": {
            "mild": 25,
            "medium": 75,
            "severe": 130
        },
        "underexposure": {
            "mild": 25,
            "medium": 90,
            "severe": 140
        },
        "occlusion": {
            "mild": 30,
            "medium": 80,
            "severe": 140
        }
    }
    
    run_batch_conversion(current_working_directory, experiment_configs)