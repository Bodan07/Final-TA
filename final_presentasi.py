import os
import cv2
from datetime import datetime
from ultralytics import YOLO

def load_model():
    """Load the models for zebra cross segmentation and vehicle detection."""
    model_zebra_cross = YOLO("E:\\mil TA\\weight_v2\\811_yolov9e_100ep_001lr_AdamW.pt")
    model_kendaraan = YOLO("yolov9e.pt")
    return model_zebra_cross, model_kendaraan


def save_violation_image(frame, vehicle_id, video_path):
    """Save the violation image with bounding box."""
    x1, y1, x2, y2 = vehicle_id
    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Create subdirectory based on video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    directory_path = f"pelanggaran/{video_name}"
    os.makedirs(directory_path, exist_ok=True)

    # Prepare the filename and save the image
    date_now = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
    violation_image_path = f"{directory_path}/pelanggaran_{date_now}.jpg"
    cv2.imwrite(violation_image_path, frame)
    print(f"Violation image saved: {violation_image_path}")


# with start frame
def detect_vehicles_in_zebra_cross(video_path, model_kendaraan, largest_mask, start_frame):
    """Detect if any vehicle violates the zebra cross starting from a given frame."""
    video = cv2.VideoCapture(video_path)
    violation_detected = False
    fps = video.get(cv2.CAP_PROP_FPS)
    min_violation_time = 5
    min_violation_frames = min_violation_time * fps
    tracked_vehicles = {}

    # Skip to the specified start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (640, 384))
        results = model_kendaraan.predict(frame_resized, imgsz=640, conf=0.3, classes=[2, 3, 5, 7], save=True, iou=0.9, device=0, verbose=False)

        current_frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        for result in results:
            for bbox in result.boxes:
                vehicle_bbox = bbox.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, vehicle_bbox)

                # Ensure bounding box is within the mask dimensions
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(largest_mask.shape[1], x2), min(largest_mask.shape[0], y2)

                # Calculate overlap
                vehicle_mask_area = largest_mask[y1:y2, x1:x2]
                overlap_count = vehicle_mask_area.sum()
                total_vehicle_area = (x2 - x1) * (y2 - y1)
                
                if total_vehicle_area > 0:
                    overlap_percentage = overlap_count / total_vehicle_area
                else:
                    overlap_percentage = 0

                vehicle_id = (x1, y1, x2, y2)

                if overlap_percentage > 0.1:
                    if vehicle_id not in tracked_vehicles:
                        tracked_vehicles[vehicle_id] = {
                            "start_frame": current_frame_number,
                            "last_frame": current_frame_number,
                            "violation_saved": False
                        }
                    tracked_vehicles[vehicle_id]["last_frame"] = current_frame_number

                    # Check if violation duration exceeds minimum time and hasn't been saved
                    if (current_frame_number - tracked_vehicles[vehicle_id]["start_frame"] >= min_violation_frames and
                            not tracked_vehicles[vehicle_id]["violation_saved"]):
                        violation_detected = True
                        
                        # Save the violation image and mark it as saved
                        save_violation_image(frame_resized, vehicle_id, video_path)
                        tracked_vehicles[vehicle_id]["violation_saved"] = True  # Mark as saved
                        break

            # if violation_detected:
            #     break

    video.release()
    return violation_detected


def get_largest_zebra_cross_mask(video_path, model_zebra_cross, output_video_path="predicted_output.mp4"):
    """Get the largest zebra cross mask from the first 10 seconds of the video and save the frames as a video."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    max_frames = int(fps * 10)  # Process only the first 10 seconds

    # Initialize video writer to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    largest_mask = None
    largest_area = 0
    frame_count = 0

    while video.isOpened() and frame_count < max_frames:
        ret, frame = video.read()
        if not ret:
            break

        # Resize frame for YOLO processing
        frame_resized = cv2.resize(frame, (640, 384))
        results = model_zebra_cross.predict(frame_resized, imgsz=640, conf=0.6, device=0, verbose=False, save=True)

        # Extract masks and determine the largest zebra cross mask
        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                for mask_data in result.masks.data:
                    mask = mask_data.cpu().numpy().squeeze()
                    mask_area = mask.sum()

                    if mask_area > largest_area:
                        largest_mask = mask
                        largest_area = mask_area

        # Draw results directly on the frame
        annotated_frame = results[0].plot()  # Get the annotated frame from YOLO
        annotated_frame = cv2.resize(annotated_frame, (width, height))  # Resize back to original size

        # Write the annotated frame to the output video
        video_writer.write(annotated_frame)

        frame_count += 1

    video.release()
    video_writer.release()  # Finalize the output video
    print(f"Predicted frames saved to video: {output_video_path}")
    return largest_mask

def classify_video(video_path):
    """Classify the video and return True if a violation is detected, else False."""
    model_zebra_cross, model_kendaraan = load_model()
    output_video_path = "output_zebra_cross.mp4"
    largest_mask = get_largest_zebra_cross_mask(video_path, model_zebra_cross, output_video_path)


    if largest_mask is None:
        print(f"No zebra cross detected in video: {video_path}")
        return False

    # Calculate the frame to start vehicle detection (after 10 seconds)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    start_frame = int(fps * 10)  # Start from frame after 10 seconds

    violation_detected = detect_vehicles_in_zebra_cross(video_path, model_kendaraan, largest_mask, start_frame)
    return violation_detected

test_video_path = "videos_test_v4/pelanggaran/p (42).mp4" #1, 26, 42
result = classify_video(test_video_path)
print(f"Violation detected: {result}")