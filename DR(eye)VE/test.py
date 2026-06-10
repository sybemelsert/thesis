import cv2
import os
import numpy as np


def parse_gaze_txt(txt_path):
    """Parses frame indexes and gaze coordinates from the text file."""
    gaze_dict = {}
    if not os.path.exists(txt_path):
        print(f"Error: Text file not found at {txt_path}")
        return gaze_dict

    with open(txt_path, "r") as file:
        for line in file:
            line_str = line.strip()
            if not line_str:
                continue
                
            parts = line_str.split(",") if "," in line_str else line_str.split()
            try:
                frame_idx = int(float(parts[0]))
                gaze_x = float(parts[2])
                gaze_y = float(parts[3])
                
                # Filter out corrupt tracker values
                if np.isnan(gaze_x) or np.isnan(gaze_y) or np.isinf(gaze_x) or np.isinf(gaze_y):
                    continue
                    
                gaze_dict[frame_idx] = (gaze_x, gaze_y)
            except (ValueError, IndexError):
                continue
    return gaze_dict


def visualize_gaze(video_path, txt_path, output_path):
    """Overlays a black gaze point circle onto the video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    gaze_dict = parse_gaze_txt(txt_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize tracking memory to handle dropped frames smoothly
    last_gaze_x, last_gaze_y = width // 2, height // 2
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video. Output will save to: {output_path}")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Coordinate lookup and adjustment math
        if frame_idx in gaze_dict:
            gaze_x, gaze_y = gaze_dict[frame_idx]
            old_width, old_height = 1280, 720
            
            # Apply your specific inversion and scaling math
            new_gaze_x = width - ((gaze_x / old_width) * width)
            new_gaze_y = height - ((gaze_y / old_height) * height)
            
            # Constrain pixels safely inside frame boundaries
            new_gaze_x = min(width - 1, max(0, int(new_gaze_x)))
            new_gaze_y = min(height - 1, max(0, int(new_gaze_y)))
            
            last_gaze_x, last_gaze_y = new_gaze_x, new_gaze_y
        else:
            # Dropback fallback position
            new_gaze_x, new_gaze_y = last_gaze_x, last_gaze_y
            
        # 2. Draw the small black circle
        # cv2.circle params: (target_image, center_coordinates, radius_pixels, color_bgr, thickness)
        # thickness=-1 completely fills the circle with the color
        cv2.circle(frame, (new_gaze_x, new_gaze_y), radius=12, color=(0, 0, 0), thickness=-1)
        
        out_writer.write(frame)
        frame_idx += 1
        
    cap.release()
    out_writer.release()
    print("Gaze rendering complete!")


if __name__ == "__main__":
    # Configure your paths here
    video_file = "01.avi"
    text_file = "01.txt"
    output_file = "01_gaze_visualization.avi"
    
    visualize_gaze(video_file, text_file, output_file)