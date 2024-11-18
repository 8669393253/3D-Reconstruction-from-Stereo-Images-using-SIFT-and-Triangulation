import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect and match keypoints using SIFT
def detect_and_match_features(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN based matcher
    index_params = dict(algorithm=1, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter out bad matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches, des1, des2

# Load two images from different angles
img1 = cv2.imread(r"D:\Desktop\1.jpg")
img2 = cv2.imread(r"D:\Desktop\2.jpg")

# Detect and match features
kp1, kp2, good_matches, des1, des2 = detect_and_match_features(img1, img2)

# Visualize the matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(10, 5))
plt.imshow(img_matches)
plt.title("Feature Matches")
plt.show()

# Dummy camera matrix (intrinsic parameters of the camera)
focal_length = 800  # Example focal length in pixels
center = (640, 360)  # Principal point (image center)
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]])

# Distortion coefficients (assuming no distortion)
dist_coeffs = np.zeros((4, 1))

def triangulate_points(K, kp1, kp2, matches, img1, img2):
    # Get corresponding points in both images
    points1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    points2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Create a dummy homography matrix (you would usually find this from fundamental matrix, etc.)
    # Here, for simplicity, we'll just assume we know the relative camera position
    # In practice, use stereo camera calibration to find relative transformation (R, T)
    R = np.eye(3)  # Identity matrix for simplicity
    T = np.array([[1], [0], [0]])  # Arbitrary translation vector (for illustration)

    # Construct projection matrices for both cameras (using camera matrices K)
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))  # Camera 1 projection matrix
    P2 = np.dot(K, np.hstack((R, T)))  # Camera 2 projection matrix

    # Triangulate points
    points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

    # Convert homogeneous coordinates to 3D
    points3D = points4D / points4D[3]  # Normalize by the homogeneous coordinate
    points3D = points3D[:3].T  # Drop the homogeneous coordinate

    return points3D

# Triangulate 3D points
points3D = triangulate_points(K, kp1, kp2, good_matches, img1, img2)

# Visualize the 3D points
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c='r', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D Reconstruction")
plt.show()

