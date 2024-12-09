import cv2
import tensorflow as tf
import posenet


def draw_skeleton(image, keypoints, threshold=0.2):
    skeleton_points = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_hip': 11,
        'right_hip': 12,
    }

    # Draw torso skeleton lines
    torso_pairs = [
        ('left_shoulder', 'right_shoulder'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),
        ('left_hip', 'right_hip'),
    ]

    for pair in torso_pairs:
        part1, part2 = skeleton_points[pair[0]], skeleton_points[pair[1]]
        if keypoints[part1][2] > threshold and keypoints[part2][2] > threshold:
            point1 = (int(keypoints[part1][1]), int(keypoints[part1][0]))
            point2 = (int(keypoints[part2][1]), int(keypoints[part2][0]))
            cv2.line(image, point1, point2, (0, 255, 0), 2)

    # Draw keypoints for the torso
    for point in skeleton_points.values():
        if keypoints[point][2] > threshold:
            center = (int(keypoints[point][1]), int(keypoints[point][0]))
            cv2.circle(image, center, 5, (0, 0, 255), -1)

    return image


def main():
    model = posenet.load_model(101)  # Load PoseNet model without a session in TF2

    cap = cv2.VideoCapture(0)  # Use the default webcam
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_image, display_image, output_scale = posenet.process_input(frame, scale_factor=0.7125,
                                                                             output_stride=16)
            # Run the model directly without sess.run()
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result,
                output_stride=16, max_pose_detections=1, min_pose_score=0.25
            )
            keypoint_coords *= output_scale

            for pi in range(len(pose_scores)):
                if pose_scores[pi] > 0.2:
                    display_image = draw_skeleton(display_image, keypoint_coords[pi])

            cv2.imshow('PoseNet - Torso Detection', display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
