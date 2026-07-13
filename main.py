import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import tkinter as tk
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# ==========================================================
# DATASET PATH
# ==========================================================
dataset_path = r"archive (2)"

# ==========================================================
# OUTPUT FOLDER
# ==========================================================
output_folder = "Pose_Output_Frames"
os.makedirs(output_folder, exist_ok=True)

# ==========================================================
# MEDIAPIPE BLAZEPOSE
# ==========================================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# DISPLAY CONTROL
# ==========================================================
samples_per_class = 3
display_count = {}

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1]-b[1],
        c[0]-b[0]
    ) - np.arctan2(
        a[1]-b[1],
        a[0]-b[0]
    )

    angle = np.abs(
        radians * 180.0 / np.pi
    )

    if angle > 180:
        angle = 360 - angle

    return angle


def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )

# ==========================================================
# FEATURE STORAGE
# ==========================================================
joint_angles_all = []
joint_distances_all = []
limb_lengths_all = []

alignment_scores = []
symmetry_scores = []
orientation_angles = []

velocity_values = []
acceleration_values = []

trajectory_x = []
trajectory_y = []

prev_center = None
prev_velocity = None

# ==========================================================
# REFERENCE MOVEMENT STORAGE (PER CLASS)
# ==========================================================
class_joint_angles = defaultdict(list)
class_shoulder_dist = defaultdict(list)
class_upper_arm = defaultdict(list)

class_left_shoulder = defaultdict(list)
class_right_shoulder = defaultdict(list)
class_left_elbow = defaultdict(list)
class_left_wrist = defaultdict(list)
class_left_hip = defaultdict(list)
class_right_hip = defaultdict(list)
class_left_knee = defaultdict(list)
class_right_knee = defaultdict(list)
class_left_ankle = defaultdict(list)

class_trajectory_x = defaultdict(list)
class_trajectory_y = defaultdict(list)

class_alignment = defaultdict(list)
class_symmetry = defaultdict(list)
class_orientation = defaultdict(list)

class_right_elbow_angle = defaultdict(list)
class_left_knee_angle = defaultdict(list)
class_right_knee_angle = defaultdict(list)
class_left_hip_angle = defaultdict(list)
class_right_hip_angle = defaultdict(list)
class_stride_length = defaultdict(list)
class_left_arm_swing = defaultdict(list)
class_right_arm_swing = defaultdict(list)
class_norm_shoulder_distance = defaultdict(list)
class_norm_upper_arm = defaultdict(list)

# ==========================================================
# PROCESS VIDEOS
# ==========================================================
for root, dirs, files in os.walk(dataset_path):

    class_name = os.path.basename(root)

    if class_name not in display_count:
        display_count[class_name] = 0

    for file in files:

        if not file.lower().endswith(".avi"):
            continue

        video_path = os.path.join(root, file)

        print(f"Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            continue

        video_name = os.path.splitext(file)[0]

        save_video_folder = os.path.join(
            output_folder,
            class_name,
            video_name
        )

        os.makedirs(save_video_folder, exist_ok=True)

        # --------------------------------------------------
        # Total Frames
        # --------------------------------------------------
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames < 20:
            frame_indices = range(total_frames)
        else:
            frame_indices = np.linspace(
                0,
                total_frames - 1,
                20,
                dtype=int
            )

        sample_saved = False

        # --------------------------------------------------
        # Extract Only 20 Frames
        # --------------------------------------------------
        for save_id, frame_idx in enumerate(frame_indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()

            if not ret:
                continue

            # Resize
            frame = cv2.resize(
                frame,
                (640, 480),
                interpolation=cv2.INTER_LINEAR
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Pose Estimation
            results = pose.process(rgb)

            output_frame = frame.copy()

            if results.pose_landmarks:

                mp_draw.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        thickness=2,
                        circle_radius=2
                    ),
                    mp_draw.DrawingSpec(
                        thickness=2
                    )
                )

                # ================================================
                # FEATURE EXTRACTION
                # ================================================
                landmarks = results.pose_landmarks.landmark

                h, w, _ = output_frame.shape

                # ------------------------------------------------
                # Important Landmarks
                # ------------------------------------------------
                left_shoulder = (
                    int(landmarks[11].x*w),
                    int(landmarks[11].y*h)
                )

                right_shoulder = (
                    int(landmarks[12].x*w),
                    int(landmarks[12].y*h)
                )

                left_elbow = (
                    int(landmarks[13].x*w),
                    int(landmarks[13].y*h)
                )

                right_elbow = (
                    int(landmarks[14].x*w),
                    int(landmarks[14].y*h)
                )

                left_wrist = (
                    int(landmarks[15].x*w),
                    int(landmarks[15].y*h)
                )

                right_wrist = (
                    int(landmarks[16].x*w),
                    int(landmarks[16].y*h)
                )

                left_hip = (
                    int(landmarks[23].x*w),
                    int(landmarks[23].y*h)
                )

                right_hip = (
                    int(landmarks[24].x*w),
                    int(landmarks[24].y*h)
                )

                left_knee = (
                    int(landmarks[25].x*w),
                    int(landmarks[25].y*h)
                )

                right_knee = (
                    int(landmarks[26].x*w),
                    int(landmarks[26].y*h)
                )

                left_ankle = (
                    int(landmarks[27].x*w),
                    int(landmarks[27].y*h)
                )

                right_ankle = (
                    int(landmarks[28].x*w),
                    int(landmarks[28].y*h)
                )

                # ------------------------------------------------
                # GEOMETRIC FEATURES
                # ------------------------------------------------

                # Joint Angle
                elbow_angle = calculate_angle(
                    left_shoulder,
                    left_elbow,
                    left_wrist
                )

                joint_angles_all.append(elbow_angle)

                # Joint Distance
                shoulder_distance = distance(
                    left_shoulder,
                    right_shoulder
                )

                joint_distances_all.append(
                    shoulder_distance
                )

                # Limb Length
                upper_arm = distance(
                    left_shoulder,
                    left_elbow
                )

                limb_lengths_all.append(
                    upper_arm
                )

                # ------------------------------------------------
                # EXTRA DISCRIMINATIVE FEATURES
                # ------------------------------------------------

                # Torso Length (for scale normalization)
                mid_shoulder = (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                )

                mid_hip = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )

                torso_length = distance(mid_shoulder, mid_hip)
                if torso_length < 1e-6:
                    torso_length = 1e-6

                # Right Elbow Angle
                right_elbow_angle = calculate_angle(
                    right_shoulder,
                    right_elbow,
                    right_wrist
                )

                # Left / Right Knee Angle
                left_knee_angle = calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )

                right_knee_angle = calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )

                # Left / Right Hip Angle
                left_hip_angle = calculate_angle(
                    left_shoulder,
                    left_hip,
                    left_knee
                )

                right_hip_angle = calculate_angle(
                    right_shoulder,
                    right_hip,
                    right_knee
                )

                # Stride Length (ankle-to-ankle, normalized)
                stride_length = distance(left_ankle, right_ankle) / torso_length

                # Arm Swing (wrist-to-hip distance, normalized) - useful for boxing
                left_arm_swing = distance(left_wrist, left_hip) / torso_length
                right_arm_swing = distance(right_wrist, right_hip) / torso_length

                # Normalized Core Distances
                norm_shoulder_distance = shoulder_distance / torso_length
                norm_upper_arm = upper_arm / torso_length

                # ------------------------------------------------
                # SPATIAL FEATURES
                # ------------------------------------------------

                # Skeleton Alignment
                alignment = abs(
                    left_shoulder[1]
                    -
                    right_shoulder[1]
                )

                alignment_scores.append(
                    alignment
                )

                # Body Symmetry
                left_leg = distance(
                    left_hip,
                    left_knee
                )

                right_leg = distance(
                    right_hip,
                    right_knee
                )

                symmetry = abs(
                    left_leg - right_leg
                )

                symmetry_scores.append(
                    symmetry
                )

                # Orientation
                orientation = np.degrees(
                    np.arctan2(
                        right_shoulder[1]
                        -
                        left_shoulder[1],
                        right_shoulder[0]
                        -
                        left_shoulder[0]
                    )
                )

                orientation_angles.append(
                    orientation
                )

                # ------------------------------------------------
                # MOTION FEATURES
                # ------------------------------------------------

                body_center = (
                    int(
                        (
                            left_hip[0]
                            +
                            right_hip[0]
                        )/2
                    ),
                    int(
                        (
                            left_hip[1]
                            +
                            right_hip[1]
                        )/2
                    )
                )

                trajectory_x.append(
                    body_center[0]
                )

                trajectory_y.append(
                    body_center[1]
                )

                if prev_center is not None:

                    velocity = distance(
                        body_center,
                        prev_center
                    )

                    velocity_values.append(
                        velocity
                    )

                    if prev_velocity is not None:

                        acceleration = (
                            velocity
                            -
                            prev_velocity
                        )

                        acceleration_values.append(
                            acceleration
                        )

                    prev_velocity = velocity

                prev_center = body_center

                # ------------------------------------------------
                # PRINT FEATURES
                # ------------------------------------------------
                print(
                    f"Angle:{elbow_angle:.2f} | "
                    f"Distance:{shoulder_distance:.2f} | "
                    f"Limb:{upper_arm:.2f}"
                )

                # ------------------------------------------------
                # STORE PER-CLASS DATA FOR REFERENCE MOVEMENT
                # ------------------------------------------------
                class_joint_angles[class_name].append(elbow_angle)
                class_shoulder_dist[class_name].append(shoulder_distance)
                class_upper_arm[class_name].append(upper_arm)

                class_left_shoulder[class_name].append(left_shoulder)
                class_right_shoulder[class_name].append(right_shoulder)
                class_left_elbow[class_name].append(left_elbow)
                class_left_wrist[class_name].append(left_wrist)
                class_left_hip[class_name].append(left_hip)
                class_right_hip[class_name].append(right_hip)
                class_left_knee[class_name].append(left_knee)
                class_right_knee[class_name].append(right_knee)
                class_left_ankle[class_name].append(left_ankle)

                class_trajectory_x[class_name].append(body_center[0])
                class_trajectory_y[class_name].append(body_center[1])

                class_alignment[class_name].append(alignment)
                class_symmetry[class_name].append(symmetry)
                class_orientation[class_name].append(orientation)

                class_right_elbow_angle[class_name].append(right_elbow_angle)
                class_left_knee_angle[class_name].append(left_knee_angle)
                class_right_knee_angle[class_name].append(right_knee_angle)
                class_left_hip_angle[class_name].append(left_hip_angle)
                class_right_hip_angle[class_name].append(right_hip_angle)
                class_stride_length[class_name].append(stride_length)
                class_left_arm_swing[class_name].append(left_arm_swing)
                class_right_arm_swing[class_name].append(right_arm_swing)
                class_norm_shoulder_distance[class_name].append(norm_shoulder_distance)
                class_norm_upper_arm[class_name].append(norm_upper_arm)

            # ------------------------------------------
            # Save Frame
            # ------------------------------------------
            save_path = os.path.join(
                save_video_folder,
                f"frame_{save_id:03d}.jpg"
            )

            cv2.imwrite(save_path, output_frame)

            # ------------------------------------------
            # Display 3 Samples Per Class
            # ------------------------------------------
            if (
                display_count[class_name] < samples_per_class
                and not sample_saved
                and results.pose_landmarks
            ):

                plt.figure(figsize=(12,6))

                plt.suptitle(
                    f"{class_name}",
                    fontsize=18,
                    fontweight="bold"
                )

                plt.subplot(1,2,1)
                plt.imshow(cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                ))
                plt.title(
                    "Original Frame",
                    fontweight="bold"
                )
                plt.axis("off")

                plt.subplot(1,2,2)
                plt.imshow(cv2.cvtColor(
                    output_frame,
                    cv2.COLOR_BGR2RGB
                ))
                plt.title(
                    "MediaPipe BlazePose",
                    fontweight="bold"
                )
                plt.axis("off")

                plt.tight_layout()
                plt.show()

                display_count[class_name] += 1
                sample_saved = True

        cap.release()

        print(
            f"{video_name} --> "
            f"{len(frame_indices)} frames saved"
        )

pose.close()

print("\nCompleted Successfully")
print("Output Folder:", output_folder)

# ==========================================================
# PLOT RESULTS
# ==========================================================

# Joint Angles
plt.figure(figsize=(8,5))
plt.plot(joint_angles_all)
plt.title("Joint Angles")
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.grid(True)
plt.show()

# Joint Distance
plt.figure(figsize=(8,5))
plt.plot(joint_distances_all)
plt.title("Joint Distance")
plt.xlabel("Frame")
plt.ylabel("Pixels")
plt.grid(True)
plt.show()

# Limb Length
plt.figure(figsize=(8,5))
plt.plot(limb_lengths_all)
plt.title("Limb Length")
plt.xlabel("Frame")
plt.ylabel("Pixels")
plt.grid(True)
plt.show()

# Skeleton Alignment
plt.figure(figsize=(8,5))
plt.plot(alignment_scores)
plt.title("Skeleton Alignment Error")
plt.xlabel("Frame")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# Body Symmetry
plt.figure(figsize=(8,5))
plt.plot(symmetry_scores)
plt.title("Body Symmetry Error")
plt.xlabel("Frame")
plt.ylabel("Error")
plt.grid(True)
plt.show()

# Body Orientation
plt.figure(figsize=(8,5))
plt.plot(orientation_angles)
plt.title("Body Orientation")
plt.xlabel("Frame")
plt.ylabel("Degrees")
plt.grid(True)
plt.show()

# Velocity
plt.figure(figsize=(8,5))
plt.plot(velocity_values)
plt.title("Velocity")
plt.xlabel("Frame")
plt.ylabel("Velocity")
plt.grid(True)
plt.show()

# Acceleration
plt.figure(figsize=(8,5))
plt.plot(acceleration_values)
plt.title("Acceleration")
plt.xlabel("Frame")
plt.ylabel("Acceleration")
plt.grid(True)
plt.show()

# Movement Trajectory
plt.figure(figsize=(8,6))
plt.plot(
    trajectory_x,
    trajectory_y,
    marker='o'
)
plt.title(
    "Movement Trajectory"
)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# ==========================================================
# REFERENCE MOVEMENT GENERATION (NOVEL MODULE)
# ==========================================================
print("\n==================================================")
print("REFERENCE MOVEMENT GENERATION")
print("==================================================")

reference_movement_library = {}

for class_name in class_joint_angles.keys():

    # ------------------------------------------------------
    # 1. STANDARD POSE GENERATION
    # ------------------------------------------------------
    standard_pose = {
        "left_shoulder": tuple(np.mean(class_left_shoulder[class_name], axis=0)),
        "right_shoulder": tuple(np.mean(class_right_shoulder[class_name], axis=0)),
        "left_elbow": tuple(np.mean(class_left_elbow[class_name], axis=0)),
        "left_wrist": tuple(np.mean(class_left_wrist[class_name], axis=0)),
        "left_hip": tuple(np.mean(class_left_hip[class_name], axis=0)),
        "right_hip": tuple(np.mean(class_right_hip[class_name], axis=0)),
        "left_knee": tuple(np.mean(class_left_knee[class_name], axis=0)),
        "right_knee": tuple(np.mean(class_right_knee[class_name], axis=0)),
        "left_ankle": tuple(np.mean(class_left_ankle[class_name], axis=0)),
    }

    # ------------------------------------------------------
    # 2. MEAN JOINT ANGLE COMPUTATION
    # ------------------------------------------------------
    mean_elbow_angle = np.mean(class_joint_angles[class_name])
    mean_shoulder_distance = np.mean(class_shoulder_dist[class_name])
    mean_upper_arm = np.mean(class_upper_arm[class_name])

    # ------------------------------------------------------
    # 3. REFERENCE SKELETON CONSTRUCTION
    # ------------------------------------------------------
    reference_skeleton = [
        standard_pose["left_shoulder"],
        standard_pose["right_shoulder"],
        standard_pose["left_elbow"],
        standard_pose["left_wrist"],
        standard_pose["left_hip"],
        standard_pose["right_hip"],
        standard_pose["left_knee"],
        standard_pose["right_knee"],
        standard_pose["left_ankle"],
    ]

    reference_skeleton_connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
    ]

    # ------------------------------------------------------
    # 4. REFERENCE MOTION TEMPLATE GENERATION
    # ------------------------------------------------------
    ref_traj_x = np.array(class_trajectory_x[class_name])
    ref_traj_y = np.array(class_trajectory_y[class_name])

    template_length = 20

    if len(ref_traj_x) >= 2:
        sample_points = np.linspace(0, len(ref_traj_x) - 1, template_length)
        motion_template_x = np.interp(sample_points, np.arange(len(ref_traj_x)), ref_traj_x)
        motion_template_y = np.interp(sample_points, np.arange(len(ref_traj_y)), ref_traj_y)
    else:
        motion_template_x = ref_traj_x
        motion_template_y = ref_traj_y

    # ------------------------------------------------------
    # STORE IN REFERENCE MOVEMENT LIBRARY
    # ------------------------------------------------------
    reference_movement_library[class_name] = {
        "standard_pose": standard_pose,
        "mean_elbow_angle": mean_elbow_angle,
        "mean_shoulder_distance": mean_shoulder_distance,
        "mean_upper_arm": mean_upper_arm,
        "reference_skeleton": reference_skeleton,
        "reference_skeleton_connections": reference_skeleton_connections,
        "motion_template_x": motion_template_x,
        "motion_template_y": motion_template_y,
    }

    # ------------------------------------------------------
    # COMMAND WINDOW OUTPUT
    # ------------------------------------------------------
    print(f"\nClass: {class_name}")
    print(f"  Mean Elbow Angle       : {mean_elbow_angle:.2f}")
    print(f"  Mean Shoulder Distance : {mean_shoulder_distance:.2f}")
    print(f"  Mean Upper Arm Length  : {mean_upper_arm:.2f}")
    print("  Standard Pose (Reference Skeleton Joints):")
    for joint_name, joint_xy in standard_pose.items():
        print(f"    {joint_name:15s}: ({joint_xy[0]:.1f}, {joint_xy[1]:.1f})")

    # ------------------------------------------------------
    # PLOT: REFERENCE SKELETON
    # ------------------------------------------------------
    joint_positions = {
        "left_shoulder": standard_pose["left_shoulder"],
        "right_shoulder": standard_pose["right_shoulder"],
        "left_elbow": standard_pose["left_elbow"],
        "left_wrist": standard_pose["left_wrist"],
        "left_hip": standard_pose["left_hip"],
        "right_hip": standard_pose["right_hip"],
        "left_knee": standard_pose["left_knee"],
        "right_knee": standard_pose["right_knee"],
        "left_ankle": standard_pose["left_ankle"],
    }

    plt.figure(figsize=(6,8))

    for j1, j2 in reference_skeleton_connections:
        x_vals = [joint_positions[j1][0], joint_positions[j2][0]]
        y_vals = [joint_positions[j1][1], joint_positions[j2][1]]
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)

    for joint_name, (jx, jy) in joint_positions.items():
        plt.scatter(jx, jy, c='red', s=60, zorder=5)

    plt.gca().invert_yaxis()
    plt.title(f"Reference Skeleton - {class_name}", fontweight="bold")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------
    # PLOT: MEAN JOINT ANGLE BAR CHART
    # ------------------------------------------------------
    plt.figure(figsize=(6,5))
    plt.bar(
        ["Elbow Angle", "Shoulder Distance", "Upper Arm Length"],
        [mean_elbow_angle, mean_shoulder_distance, mean_upper_arm],
        color=["orange", "green", "purple"]
    )
    plt.title(f"Mean Joint Features - {class_name}", fontweight="bold")
    plt.ylabel("Value")
    plt.grid(True, axis='y')
    plt.show()

    # ------------------------------------------------------
    # PLOT: REFERENCE MOTION TEMPLATE
    # ------------------------------------------------------
    plt.figure(figsize=(8,6))
    plt.plot(
        motion_template_x,
        motion_template_y,
        marker='o',
        color='darkred'
    )
    plt.gca().invert_yaxis()
    plt.title(f"Reference Motion Template - {class_name}", fontweight="bold")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

print("\n==================================================")
print("REFERENCE MOVEMENT LIBRARY GENERATED")
print("==================================================")
print("Classes stored:", list(reference_movement_library.keys()))

# ==========================================================
# HYBRID RANDOM FOREST - XGBOOST CLASSIFIER
# ==========================================================
print("\n==================================================")
print("HYBRID RANDOM FOREST - XGBOOST CLASSIFICATION")
print("==================================================")

# ----------------------------------------------------------
# BUILD FEATURE MATRIX (X) AND LABELS (y)
# ----------------------------------------------------------
X = []
y = []

for class_name in class_joint_angles.keys():

    n_samples = len(class_joint_angles[class_name])

    for i in range(n_samples):

        feature_vector = [
            class_joint_angles[class_name][i],
            class_right_elbow_angle[class_name][i],
            class_left_knee_angle[class_name][i],
            class_right_knee_angle[class_name][i],
            class_left_hip_angle[class_name][i],
            class_right_hip_angle[class_name][i],
            class_norm_shoulder_distance[class_name][i],
            class_norm_upper_arm[class_name][i],
            class_alignment[class_name][i],
            class_symmetry[class_name][i],
            class_orientation[class_name][i],
            class_stride_length[class_name][i],
            class_left_arm_swing[class_name][i],
            class_right_arm_swing[class_name][i],
        ]

        X.append(feature_vector)
        y.append(class_name)

X = np.array(X)
y = np.array(y)

print(f"Total Samples : {X.shape[0]}")
print(f"Feature Dim   : {X.shape[1]}")
print(f"Classes       : {sorted(set(y))}")

# ----------------------------------------------------------
# LABEL ENCODING
# ----------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

class_labels = label_encoder.classes_

# ----------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"Train Samples : {X_train.shape[0]}")
print(f"Test Samples  : {X_test.shape[0]}")

# ----------------------------------------------------------
# FEATURE SCALING
# ----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------------
# RANDOM FOREST CLASSIFIER
# ----------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy : {rf_accuracy:.4f}")

# ----------------------------------------------------------
# XGBOOST CLASSIFIER
# ----------------------------------------------------------
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"XGBoost Accuracy       : {xgb_accuracy:.4f}")

# ----------------------------------------------------------
# HYBRID MODEL (STACKING ENSEMBLE: RF + XGBOOST -> LOGISTIC REGRESSION)
# ----------------------------------------------------------
hybrid_model = StackingClassifier(
    estimators=[
        ("random_forest", rf_model),
        ("xgboost", xgb_model)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    passthrough=True,
    cv=5,
    n_jobs=-1
)

hybrid_model.fit(X_train, y_train)
hybrid_pred = hybrid_model.predict(X_test)
hybrid_accuracy = accuracy_score(y_test, hybrid_pred)

print(f"Hybrid RF-XGBoost Accuracy : {hybrid_accuracy:.4f}")

# ----------------------------------------------------------
# CROSS-VALIDATION CHECK (STRATIFIED 5-FOLD)
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(hybrid_model, X_train, y_train, cv=cv, n_jobs=-1)

print(f"Cross-Validation Accuracy  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ----------------------------------------------------------
# CLASSIFICATION REPORT
# ----------------------------------------------------------
print("\nHybrid Model Classification Report:")
print(
    classification_report(
        y_test,
        hybrid_pred,
        target_names=class_labels
    )
)

# ----------------------------------------------------------
# CONFUSION MATRIX
# ----------------------------------------------------------
conf_matrix = confusion_matrix(y_test, hybrid_pred)

print("Confusion Matrix:")
print(conf_matrix)

# ----------------------------------------------------------
# PLOT: MODEL ACCURACY COMPARISON
# ----------------------------------------------------------
plt.figure(figsize=(7,5))
plt.bar(
    ["Random Forest", "XGBoost", "Hybrid RF-XGBoost"],
    [rf_accuracy, xgb_accuracy, hybrid_accuracy],
    color=["steelblue", "seagreen", "darkorange"]
)
plt.title("Model Accuracy Comparison", fontweight="bold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.show()

# ----------------------------------------------------------
# PLOT: CONFUSION MATRIX
# ----------------------------------------------------------
plt.figure(figsize=(7,6))
plt.imshow(conf_matrix, cmap="Blues")
plt.title("Hybrid Model Confusion Matrix", fontweight="bold")
plt.colorbar()
plt.xticks(range(len(class_labels)), class_labels, rotation=45)
plt.yticks(range(len(class_labels)), class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(
            j, i, str(conf_matrix[i, j]),
            ha="center", va="center",
            color="black"
        )

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT: FEATURE IMPORTANCE (RANDOM FOREST)
# ----------------------------------------------------------
feature_names = [
    "Left Elbow Angle",
    "Right Elbow Angle",
    "Left Knee Angle",
    "Right Knee Angle",
    "Left Hip Angle",
    "Right Hip Angle",
    "Norm Shoulder Dist",
    "Norm Upper Arm",
    "Skeleton Alignment",
    "Body Symmetry",
    "Orientation",
    "Stride Length",
    "Left Arm Swing",
    "Right Arm Swing"
]

plt.figure(figsize=(8,6))
plt.bar(feature_names, rf_model.feature_importances_, color="purple")
plt.title("Random Forest Feature Importance", fontweight="bold")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

print("\n==================================================")
print("CLASSIFICATION COMPLETED SUCCESSFULLY")
print("==================================================")

# ==========================================================
# ADDITIONAL EVALUATION PLOTS
# ==========================================================

hybrid_proba = hybrid_model.predict_proba(X_test)

n_classes = len(class_labels)

y_test_bin = label_binarize(
    y_test,
    classes=list(range(n_classes))
)

# ----------------------------------------------------------
# ROC CURVE (ONE-VS-REST, MULTICLASS)
# ----------------------------------------------------------
plt.figure(figsize=(8,6))

for i in range(n_classes):

    fpr, tpr, _ = roc_curve(y_test_bin[:, i], hybrid_proba[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr,
        label=f"{class_labels[i]} (AUC = {roc_auc:.2f})"
    )

plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.title("ROC Curve - Hybrid RF-XGBoost", fontweight="bold")
plt.xlabel("False Positive Rate",fontweight='bold')
plt.ylabel("True Positive Rate",fontweight='bold')
plt.legend(loc="lower right")
plt.savefig('Roc curve.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# PRECISION-RECALL CURVE (ONE-VS-REST, MULTICLASS)
# ----------------------------------------------------------
plt.figure(figsize=(8,6))

for i in range(n_classes):

    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test_bin[:, i], hybrid_proba[:, i]
    )

    plt.plot(
        recall_vals, precision_vals,
        label=f"{class_labels[i]}"
    )

plt.title("Precision-Recall Curve - Hybrid RF-XGBoost", fontweight="bold")
plt.xlabel("Recall",fontweight='bold')
plt.ylabel("Precision",fontweight='bold')
plt.legend(loc="lower left")
plt.savefig('precision and recall.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# FPR AND FNR BAR PLOT (PER CLASS)
# ----------------------------------------------------------
fpr_list = []
fnr_list = []

for i in range(n_classes):

    tp = conf_matrix[i, i]
    fn = conf_matrix[i, :].sum() - tp
    fp = conf_matrix[:, i].sum() - tp
    tn = conf_matrix.sum() - tp - fn - fp

    fpr_class = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_class = fn / (fn + tp) if (fn + tp) > 0 else 0

    fpr_list.append(fpr_class)
    fnr_list.append(fnr_class)

    print(f"{class_labels[i]} -> FPR: {fpr_class:.4f} | FNR: {fnr_class:.4f}")

x_pos = np.arange(n_classes)
bar_width = 0.35

plt.figure(figsize=(8,6))
plt.bar(x_pos - bar_width/2, fpr_list, width=bar_width, label="FPR", color="crimson")
plt.bar(x_pos + bar_width/2, fnr_list, width=bar_width, label="FNR", color="royalblue")
plt.xticks(x_pos, class_labels)
plt.title("False Positive Rate & False Negative Rate per Class", fontweight="bold")
plt.ylabel("Rate",fontweight='bold')
plt.xlabel("Class",fontweight='bold')
plt.legend()
plt.savefig('FPR and Fnr.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# PERFORMANCE METRICS (ACCURACY, PRECISION, RECALL, F1)
# ----------------------------------------------------------
overall_precision = precision_score(y_test, hybrid_pred, average="weighted")
overall_recall = recall_score(y_test, hybrid_pred, average="weighted")
overall_f1 = f1_score(y_test, hybrid_pred, average="weighted")

print(f"\nOverall Accuracy  : {hybrid_accuracy:.4f}")
print(f"Overall Precision : {overall_precision:.4f}")
print(f"Overall Recall    : {overall_recall:.4f}")
print(f"Overall F1-Score  : {overall_f1:.4f}")

plt.figure(figsize=(7,5))
plt.bar(
    ["Accuracy", "Precision", "Recall", "F1-Score"],
    [hybrid_accuracy, overall_precision, overall_recall, overall_f1],
    color=["steelblue", "seagreen", "darkorange", "purple"]
)
plt.title("Hybrid Model Performance Metrics", fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.show()

# ----------------------------------------------------------
# CALIBRATION CURVE (ONE-VS-REST, MULTICLASS)
# ----------------------------------------------------------
plt.figure(figsize=(12,8))

for i in range(n_classes):

    prob_true, prob_pred = calibration_curve(
        y_test_bin[:, i], hybrid_proba[:, i], n_bins=10
    )

    plt.plot(
        prob_pred, prob_true,
        marker='o',
        label=f"{class_labels[i]}"
    )

plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
plt.title("Calibration Curve - Hybrid RF-XGBoost", fontweight="bold")
plt.xlabel("Mean Predicted Probability",fontweight="bold")
plt.ylabel("Fraction of Positives",fontweight='bold')
plt.legend(loc="lower left")
plt.legend()
plt.savefig('Calibration Curve.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# MODEL LOSS CURVE (XGBOOST TRAIN VS VALIDATION)
# ----------------------------------------------------------
xgb_eval_results = xgb_model.evals_result()

train_loss = xgb_eval_results["validation_0"]["mlogloss"]
val_loss = xgb_eval_results["validation_1"]["mlogloss"]

plt.figure(figsize=(8,6))
plt.plot(train_loss, label="Train Loss",color='#2E4540')
plt.plot(val_loss, label="Validation Loss",color='#4B1426')
plt.title("XGBoost Model Loss Curve", fontweight="bold")
plt.xlabel("Boosting Round",fontweight='bold')
plt.ylabel("Log Loss (mlogloss)",fontweight='bold')
plt.legend()
plt.savefig('XGBoost Model Loss Curve.png',dpi=800)
plt.show()

print("\n==================================================")
print("ALL EVALUATION PLOTS GENERATED SUCCESSFULLY")
print("==================================================")

# ==========================================================
# MODULE 7 - MOVEMENT STANDARDIZATION
# ==========================================================
print("\n==================================================")
print("MOVEMENT STANDARDIZATION")
print("==================================================")

def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-9:
        return 0.0
    return np.dot(vec_a, vec_b) / denom


def compute_dtw(seq_a, seq_b):
    seq_a = np.array(seq_a)
    seq_b = np.array(seq_b)
    n = len(seq_a)
    m = len(seq_b)

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )

    return dtw_matrix[n, m]


movement_standardization_results = {}

for class_name in reference_movement_library.keys():

    ref_data = reference_movement_library[class_name]

    # --------------------------------------------------
    # REFERENCE POSE (mean joint angle vector + skeleton)
    # --------------------------------------------------
    reference_angle_vector = [
        ref_data["mean_elbow_angle"],
        np.mean(class_right_elbow_angle[class_name]),
        np.mean(class_left_knee_angle[class_name]),
        np.mean(class_right_knee_angle[class_name]),
        np.mean(class_left_hip_angle[class_name]),
        np.mean(class_right_hip_angle[class_name]),
    ]

    reference_skeleton_points = np.array(ref_data["reference_skeleton"])

    n_frames = len(class_joint_angles[class_name])

    joint_angle_deviations = []
    cosine_similarities = []
    euclidean_distances = []
    symmetry_errors = []
    alignment_errors = []

    mean_symmetry = np.mean(class_symmetry[class_name])
    mean_alignment = np.mean(class_alignment[class_name])

    for i in range(n_frames):

        # --------------------------------------------------
        # STUDENT POSE (observed frame i)
        # --------------------------------------------------
        student_angle_vector = [
            class_joint_angles[class_name][i],
            class_right_elbow_angle[class_name][i],
            class_left_knee_angle[class_name][i],
            class_right_knee_angle[class_name][i],
            class_left_hip_angle[class_name][i],
            class_right_hip_angle[class_name][i],
        ]

        student_skeleton_points = np.array([
            class_left_shoulder[class_name][i],
            class_right_shoulder[class_name][i],
            class_left_elbow[class_name][i],
            class_left_wrist[class_name][i],
            class_left_hip[class_name][i],
            class_right_hip[class_name][i],
            class_left_knee[class_name][i],
            class_right_knee[class_name][i],
            class_left_ankle[class_name][i],
        ])

        # Joint Angle Deviation
        angle_deviation = np.mean(
            np.abs(np.array(student_angle_vector) - np.array(reference_angle_vector))
        )
        joint_angle_deviations.append(angle_deviation)

        # Cosine Similarity
        cos_sim = cosine_similarity(student_angle_vector, reference_angle_vector)
        cosine_similarities.append(cos_sim)

        # Euclidean Distance (skeleton keypoints)
        eu_dist = np.mean(
            np.linalg.norm(student_skeleton_points - reference_skeleton_points, axis=1)
        )
        euclidean_distances.append(eu_dist)

        # Body Symmetry Error
        symmetry_errors.append(
            abs(class_symmetry[class_name][i] - mean_symmetry)
        )

        # Alignment Error
        alignment_errors.append(
            abs(class_alignment[class_name][i] - mean_alignment)
        )

    # --------------------------------------------------
    # DTW (student joint-angle sequence vs reference template)
    # --------------------------------------------------
    reference_angle_template = np.interp(
        np.linspace(0, n_frames - 1, 20),
        np.arange(n_frames),
        class_joint_angles[class_name]
    )

    dtw_distance = compute_dtw(
        class_joint_angles[class_name],
        reference_angle_template
    )

    # --------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------
    mean_angle_deviation = np.mean(joint_angle_deviations)
    mean_cosine_similarity = np.mean(cosine_similarities)
    mean_euclidean_distance = np.mean(euclidean_distances)
    mean_symmetry_error = np.mean(symmetry_errors)
    mean_alignment_error = np.mean(alignment_errors)

    # --------------------------------------------------
    # MOVEMENT STANDARDIZATION SCORE (0-100)
    # --------------------------------------------------
    angle_score = 100 * np.exp(-mean_angle_deviation / 30)
    similarity_score = mean_cosine_similarity * 100
    distance_score = 100 * np.exp(-mean_euclidean_distance / 50)
    symmetry_score = 100 * np.exp(-mean_symmetry_error / 30)
    alignment_score = 100 * np.exp(-mean_alignment_error / 30)
    dtw_score = 100 * np.exp(-dtw_distance / (30 * n_frames))

    movement_standardization_score = np.mean([
        angle_score,
        similarity_score,
        distance_score,
        symmetry_score,
        alignment_score,
        dtw_score
    ])

    movement_standardization_results[class_name] = {
        "joint_angle_deviation": mean_angle_deviation,
        "cosine_similarity": mean_cosine_similarity,
        "euclidean_distance": mean_euclidean_distance,
        "symmetry_error": mean_symmetry_error,
        "alignment_error": mean_alignment_error,
        "dtw_distance": dtw_distance,
        "angle_score": angle_score,
        "similarity_score": similarity_score,
        "distance_score": distance_score,
        "symmetry_score": symmetry_score,
        "alignment_score": alignment_score,
        "dtw_score": dtw_score,
        "movement_standardization_score": movement_standardization_score
    }

    print(f"\nClass: {class_name}")
    print(f"  Joint Angle Deviation : {mean_angle_deviation:.2f}")
    print(f"  Cosine Similarity     : {mean_cosine_similarity:.4f}")
    print(f"  Euclidean Distance    : {mean_euclidean_distance:.2f}")
    print(f"  Body Symmetry Error   : {mean_symmetry_error:.2f}")
    print(f"  Alignment Error       : {mean_alignment_error:.2f}")
    print(f"  DTW Distance          : {dtw_distance:.2f}")
    print(f"  Movement Standardization Score : {movement_standardization_score:.2f} / 100")

# ----------------------------------------------------------
# PLOT: RAW DEVIATION METRICS PER CLASS
# ----------------------------------------------------------
classes_list = list(movement_standardization_results.keys())

plt.figure(figsize=(10,6))

metric_keys = [
    "joint_angle_deviation",
    "euclidean_distance",
    "symmetry_error",
    "alignment_error",
    "dtw_distance"
]

x_pos = np.arange(len(classes_list))
bar_width = 0.15

for idx, metric in enumerate(metric_keys):
    values = [movement_standardization_results[c][metric] for c in classes_list]
    plt.bar(x_pos + idx * bar_width, values, width=bar_width, label=metric)

plt.xticks(x_pos + bar_width * 2, classes_list)
plt.title("Movement Standardization - Raw Deviation Metrics", fontweight="bold")
plt.ylabel("Deviation ",fontweight='bold')
plt.xlabel("Movement",fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('Movement Standardization - Raw Deviation Metrics.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# PLOT: MOVEMENT STANDARDIZATION SCORE PER CLASS
# ----------------------------------------------------------
plt.figure(figsize=(15,8))
scores = [movement_standardization_results[c]["movement_standardization_score"] for c in classes_list]
plt.bar(classes_list, scores, color="teal")
plt.title("Movement Standardization Score", fontweight="bold")
plt.ylabel("Score (0-100)",fontweight='bold')
plt.xlabel("Movement",fontweight='bold')
plt.ylim(0, 100)
plt.savefig('Movement Standardization Score.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# PLOT: COSINE SIMILARITY PER CLASS
# ----------------------------------------------------------
plt.figure(figsize=(15,8))
cos_values = [movement_standardization_results[c]["cosine_similarity"] for c in classes_list]
plt.bar(classes_list, cos_values, color="darkgoldenrod")
plt.title("Pose Cosine Similarity ", fontweight="bold")
plt.ylabel("Cosine Similarity",fontweight='bold')
plt.xlabel("Pose Cosine Similarity",fontweight='bold')
plt.ylim(0, 1)
plt.savefig('Pose Cosine Similarity.png',dpi=800)
plt.show()

print("\n==================================================")
print("MOVEMENT STANDARDIZATION COMPLETED")
print("==================================================")

# ==========================================================
# MODULE 8 - PERFORMANCE SCORING MODULE
# ==========================================================
print("\n==================================================")
print("PERFORMANCE SCORING MODULE")
print("==================================================")

performance_weights = {
    "joint_angle_accuracy": 0.35,
    "pose_similarity": 0.25,
    "body_symmetry": 0.15,
    "motion_consistency": 0.15,
    "stability": 0.10
}

performance_scoring_results = {}

for class_name in classes_list:

    results = movement_standardization_results[class_name]

    # Joint Angle Accuracy (35%)
    joint_angle_accuracy = results["angle_score"]

    # Pose Similarity (25%) - combine cosine similarity + euclidean distance score
    pose_similarity = (results["similarity_score"] + results["distance_score"]) / 2

    # Body Symmetry (15%)
    body_symmetry = results["symmetry_score"]

    # Motion Consistency (15%) - based on DTW score
    motion_consistency = results["dtw_score"]

    # Stability (10%) - based on inverse variance of joint angles + alignment score
    angle_std = np.std(class_joint_angles[class_name])
    stability_raw_score = 100 * np.exp(-angle_std / 20)
    stability = (stability_raw_score + results["alignment_score"]) / 2

    final_score = (
        joint_angle_accuracy * performance_weights["joint_angle_accuracy"] +
        pose_similarity * performance_weights["pose_similarity"] +
        body_symmetry * performance_weights["body_symmetry"] +
        motion_consistency * performance_weights["motion_consistency"] +
        stability * performance_weights["stability"]
    )

    performance_scoring_results[class_name] = {
        "joint_angle_accuracy": joint_angle_accuracy,
        "pose_similarity": pose_similarity,
        "body_symmetry": body_symmetry,
        "motion_consistency": motion_consistency,
        "stability": stability,
        "final_score": final_score
    }

    print(f"\nClass: {class_name}")
    print(f"  Joint Angle Accuracy (35%) : {joint_angle_accuracy:.2f}")
    print(f"  Pose Similarity (25%)      : {pose_similarity:.2f}")
    print(f"  Body Symmetry (15%)        : {body_symmetry:.2f}")
    print(f"  Motion Consistency (15%)   : {motion_consistency:.2f}")
    print(f"  Stability (10%)            : {stability:.2f}")
    print(f"  FINAL PERFORMANCE SCORE    : {final_score:.2f} / 100")

# ----------------------------------------------------------
# PLOT: PERFORMANCE COMPONENT BREAKDOWN (STACKED BAR)
# ----------------------------------------------------------
plt.figure(figsize=(15,8))

components = ["joint_angle_accuracy", "pose_similarity", "body_symmetry", "motion_consistency", "stability"]
weights_list = [performance_weights[c] for c in components]

bottom = np.zeros(len(classes_list))

for comp, w in zip(components, weights_list):
    contrib = [performance_scoring_results[c][comp] * w for c in classes_list]
    plt.bar(classes_list, contrib, bottom=bottom, label=f"{comp} ({int(w*100)}%)")
    bottom += np.array(contrib)

plt.title("Performance Score Breakdown by Weighted Component", fontweight="bold")
plt.ylabel("Weighted Score Contribution",fontweight='bold')
plt.xlabel("Component",fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('Performance Score Breakdown.png',dpi=800)
plt.show()

# ----------------------------------------------------------
# PLOT: FINAL PERFORMANCE SCORE PER CLASS
# ----------------------------------------------------------
plt.figure(figsize=(15,8))
final_scores = [performance_scoring_results[c]["final_score"] for c in classes_list]
plt.bar(classes_list, final_scores, color="darkgreen")
plt.title("Final Performance Score", fontweight="bold")
plt.ylabel("Score (0-100)",fontweight='bold')
plt.xlabel("Component",fontweight='bold')
plt.ylim(0, 100)
plt.savefig('Final Performance Score.png',dpi=800)
plt.show()

print("\n==================================================")
print("PERFORMANCE SCORING COMPLETED")
print("==================================================")

# ==========================================================
# MODULE 8 (CONTINUED) - INTELLIGENT FEEDBACK
# (RULE-BASED FEEDBACK GENERATION)
# ==========================================================
print("\n==================================================")
print("INTELLIGENT FEEDBACK GENERATION")
print("==================================================")

FEEDBACK_THRESHOLD = 75

def generate_feedback(scores):

    feedback = []

    if scores["joint_angle_accuracy"] < FEEDBACK_THRESHOLD:
        feedback.append("Bend your knees further.")
        feedback.append("Raise your elbows.")

    if scores["body_symmetry"] < FEEDBACK_THRESHOLD:
        feedback.append("Keep your back straight.")

    if scores["stability"] < FEEDBACK_THRESHOLD:
        feedback.append("Improve body alignment.")

    if scores["pose_similarity"] < FEEDBACK_THRESHOLD:
        feedback.append("Align your posture closer to the reference pose.")

    if scores["motion_consistency"] < FEEDBACK_THRESHOLD:
        feedback.append("Maintain a steady and consistent movement pace.")

    if not feedback:
        feedback.append("Excellent form! Keep maintaining this movement quality.")

    return feedback


feedback_results = {}

for class_name in classes_list:

    feedback_results[class_name] = generate_feedback(
        performance_scoring_results[class_name]
    )

    print(f"\nClass: {class_name}")
    for line in feedback_results[class_name]:
        print(f"  - {line}")

print("\n==================================================")
print("INTELLIGENT FEEDBACK GENERATED SUCCESSFULLY")
print("==================================================")

# ==========================================================
# TKINTER FEEDBACK DASHBOARD (LIGHT THEME)
# ==========================================================

def launch_feedback_dashboard(feedback_data):

    root = tk.Tk()
    root.title("Intelligent Feedback Dashboard")
    root.geometry("1000x750")
    root.configure(bg="#FFFFFF")

    bold_font = ("Arial", 20, "bold")
    title_font = ("Arial", 26, "bold")

    # ------------------------------------------------------
    # SCROLLABLE CONTAINER
    # ------------------------------------------------------
    canvas = tk.Canvas(root, bg="#FFFFFF", highlightthickness=0)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#FFFFFF")

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # ------------------------------------------------------
    # TITLE
    # ------------------------------------------------------
    title_label = tk.Label(
        scroll_frame,
        text="Intelligent Feedback Dashboard",
        font=title_font,
        bg="#FFFFFF",
        fg="#0B3D91"
    )
    title_label.pack(pady=25)

    # ------------------------------------------------------
    # FEEDBACK CARD PER CLASS
    # ------------------------------------------------------
    for class_name, feedback_list in feedback_data.items():

        card = tk.Frame(
            scroll_frame,
            bg="#F2F2F2",
            bd=2,
            relief="groove"
        )
        card.pack(fill="x", padx=30, pady=15)

        class_label = tk.Label(
            card,
            text=class_name.upper(),
            font=bold_font,
            bg="#F2F2F2",
            fg="#0B5394"
        )
        class_label.pack(anchor="w", padx=15, pady=10)

        for line in feedback_list:

            fb_label = tk.Label(
                card,
                text=f"\u2022 {line}",
                font=bold_font,
                bg="#F2F2F2",
                fg="#222222",
                wraplength=900,
                justify="left"
            )
            fb_label.pack(anchor="w", padx=40, pady=5)

    root.mainloop()


launch_feedback_dashboard(feedback_results)