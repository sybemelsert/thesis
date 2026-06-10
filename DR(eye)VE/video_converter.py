import cv2
import os
import glob
import numpy as np
import random


### CORE FUNCTIONS FOR VIDEO ADJUSTMENT

def apply_gaussian_blur(frame, severity:int):
    """
    Apply Gaussian blur to the frame using OpenCV's built-in function, with kernel size based on severity.
    """
    if severity == 0: return frame # If severity is 0, return the original frame without modification
    kernel_size = severity if severity % 2 != 0 else severity + 1
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def apply_motion_blur(frame, severity:int):
    """
    Apply motion blur by creating a custom kernel that simulates horizontal motion using OpenCV's filter2D function.
    """
    if severity == 0: return frame 
    kernel_size = max(1, severity) # Ensure kernel size is at least 1 to avoid errors
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(frame, -1, kernel)

def apply_overexposure(frame, severity:int):
    """
    Apply overexposure by adding a constant value to pixel intensities, with clipping to maintain valid range.
    """
    if severity == 0: return frame
    frame_float = frame.astype(np.float32) + severity
    return np.clip(frame_float, 0, 255).astype(np.uint8) # Uint8 conversion ensures pixel values remain in the valid range after processing

def apply_underexposure(frame, severity:int):
    """
    Apply underexposure by subtracting a constant value from pixel intensities, with clipping to maintain valid range.
    """
    if severity == 0: return frame
    frame_float = frame.astype(np.float32) - severity
    return np.clip(frame_float, 0, 255).astype(np.uint8)

def apply_occlusion(frame, x:int, y:int, severity:int):
    """
    Apply occlusion by masking the provided region of the frame with black pixels.
    The difference between static and dynamic occlusion is that static uses fixed coordinates
    while dynamic uses gaze coordinates that can change per frame. Both functions ensure that
    the occlusion region is safely clipped within the frame dimensions to prevent errors.
    """
    if severity == 0: return frame
    h, w, _ = frame.shape
    processed_frame = frame.copy() # Create a copy of the original frame to avoid modifying it directly
    
    # Safe clipping that prevents negative index array slicing
    x1 = min(w, max(0, int(x - severity // 2)))
    y1 = min(h, max(0, int(y - severity // 2)))
    x2 = min(w, max(0, int(x + severity // 2)))
    y2 = min(h, max(0, int(y + severity // 2)))
    
    processed_frame[y1:y2, x1:x2] = [0, 0, 0] # Set the defined region to black (occlusion)
    return processed_frame


### .txt-FILE PARSING AND VIDEO PROCESSING

def parse_gaze_txt(txt_path):
    gaze_dict = {} # Dictionary to hold frame index as key and (gaze_x, gaze_y) tuple as value
    with open(txt_path, "r") as file:
        for line in file:
            line_str = line.strip()

            if not line_str: # Skip empty lines to avoid processing errors
                continue
                
            if "," in line_str:
                parts = line_str.split(",")
            else:
                parts = line_str.split()
                
            try:
                # Extract frame indices and gaze coordinates
                frame_idx = int(float(parts[0]))
                gaze_x = float(parts[2])
                gaze_y = float(parts[3])
                
                # Ignore coordinates if they contain NaN or infinite values
                if np.isnan(gaze_x) or np.isnan(gaze_y) or np.isinf(gaze_x) or np.isinf(gaze_y):
                    continue
                    
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

    # Initialize static occlusion coordinates only once per video to ensure consistency across all tiers
    static_center_x = None
    static_center_y = None

    # Loop through each effect type
    for effect_name, severity_levels in configurations.items():
        # Loop through the three levels of severity
        for tier, severity_val in severity_levels.items():
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video target {video_path}")
                continue
                
            # Retrieve video properties for output configuration
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Generate random static occlusion coordinates once per video if not already set
            if static_center_x is None:
                static_center_x = random.randint(int(width * 0.4), int(width * 0.6))
                static_center_y = random.randint(int(height * 0.4), int(height * 0.6))
            
            # Form output filenames that include type and tier of the video
            output_filename = f"{base_name}_{effect_name}_{tier}.avi"
            output_path = os.path.join(output_dir, output_filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            print(f"    Editing -> {output_filename} (Severity level: {severity_val})...")
            
            # Set up tracking memory for dynamic occlusion
            last_gaze_x, last_gaze_y = width // 2, height // 2
            
            # Initiate loop to read and process each frame of the video
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Trigger the correct effect and severity based on the current loop iteration
                if effect_name == "gaussian_blur":
                    processed = apply_gaussian_blur(frame, severity_val)
                elif effect_name == "motion_blur":
                    processed = apply_motion_blur(frame, severity_val)
                elif effect_name == "overexposure":
                    processed = apply_overexposure(frame, severity_val)
                elif effect_name == "underexposure":
                    processed = apply_underexposure(frame, severity_val)
                elif effect_name == "occlusion_static":
                    processed = apply_occlusion(frame, static_center_x, static_center_y, severity_val)
                elif effect_name == "occlusion_dynamic":
                    if frame_idx in gaze_dict:
                        gaze_x, gaze_y = gaze_dict[frame_idx]
                        new_gaze_x = min(width - 1, max(0, int(gaze_x)))
                        new_gaze_y = min(height - 1, max(0, int(gaze_y)))
                        last_gaze_x, last_gaze_y = new_gaze_x, new_gaze_y
                    else:
                        new_gaze_x, new_gaze_y = last_gaze_x, last_gaze_y
                        
                    processed = apply_occlusion(frame, new_gaze_x, new_gaze_y, severity_val)
                else:
                    processed = frame
                    
                out_writer.write(processed)
                frame_idx += 1
                
            cap.release()
            out_writer.release()


### AUTOMATICALLY MATCH .avi AND .txt FILES IN A DIRECTORY AND PROCESS THEM

def run_batch_conversion(target_folder, experiment_configs):
    """
    Scans a directory for matching .avi and .txt file pairs and triggers processing.
    """
    avi_files = glob.glob(os.path.join(target_folder, "*.avi"))
    
    if not avi_files:
        print(f"ERROR: No .avi files found in '{target_folder}'.")
        return

    print(f"Found {len(avi_files)} video files...")
    
    pairs_found = 0
    for video_path in avi_files:
        # Scan for corresponding .txt file by matching base name
        base_without_ext = os.path.splitext(video_path)[0]
        expected_txt_path = base_without_ext + ".txt"
        
        if os.path.exists(expected_txt_path):
            pairs_found += 1
            print(f"\n[Pair #{pairs_found}] Processing base name: {os.path.basename(base_without_ext)}")
            
            # Select output folder for resulting videos, create one if pathname doesn't exist
            output_folder = os.path.join(target_folder, "adversarial_outputs")
            os.makedirs(output_folder, exist_ok=True)
            
            # Call processing function for matched pair
            process_single_pair(
                video_path=video_path,
                txt_path=expected_txt_path,
                output_dir=output_folder,
                configurations=experiment_configs
            )
        else:
            print(f"\nSkipping: Found '{os.path.basename(video_path)}' but no matching '{os.path.basename(expected_txt_path)}' exists in the same folder.")

    print(f"\nConversion complete! Total matched pairs processed: {pairs_found}")


### MAIN EXECUTION

if __name__ == "__main__":
    # Get the current working directory of the script to ensure it processes files in the correct location
    current_working_directory = os.path.dirname(os.path.abspath(__file__)) 
    
    experiment_configs = {
        "gaussian_blur": { 
            "mild": 9,
            "medium": 25,
            "severe": 55
        },
        "motion_blur": {
            "mild": 5,
            "medium": 20,
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
        "occlusion_static": {
            "mild": 35,
            "medium": 80,
            "severe": 150
        },
        "occlusion_dynamic": {
            "mild": 35,
            "medium": 80,
            "severe": 150
        }
    }
    
    # Run the converter
    run_batch_conversion(current_working_directory, experiment_configs)