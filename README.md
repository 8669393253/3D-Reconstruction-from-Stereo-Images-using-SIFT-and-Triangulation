# 3D Reconstruction from Stereo Images using SIFT and Triangulation

This project demonstrates how to perform 3D reconstruction from two images taken from different viewpoints using feature detection and matching. The core algorithms used are **SIFT (Scale-Invariant Feature Transform)** for feature detection and **FLANN (Fast Library for Approximate Nearest Neighbors)** for feature matching. The reconstructed 3D points are then visualized using **matplotlib**.

## Requirements

Before running the code, make sure you have the following Python libraries installed:

- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

You can install these libraries via pip:

pip install opencv-python numpy matplotlib


### Python Version Compatibility
This code is written in Python 3.x, and has been tested on Python 3.7+. Ensure you are using a compatible version.

## Project Overview

The code performs the following tasks:
1. **Feature Detection and Matching**: Detects keypoints in two images and matches them using the SIFT algorithm.
2. **Camera Matrix Setup**: Defines a camera matrix (intrinsic parameters) and assumes zero distortion.
3. **3D Triangulation**: Computes the 3D coordinates of matched keypoints using triangulation.
4. **Visualization**: Displays the feature matches and the 3D points using `matplotlib`.

## Code Structure

1. **`detect_and_match_features(img1, img2)`**:  
   This function takes two input images, converts them to grayscale, detects keypoints using SIFT, and matches these keypoints using FLANN. It returns the keypoints, descriptors, and good matches.

2. **`triangulate_points(K, kp1, kp2, matches, img1, img2)`**:  
   This function performs triangulation on the matched keypoints. It uses the camera matrices to project the 2D keypoints into 3D space. The code assumes a simplified relative camera transformation (identity rotation and arbitrary translation vector).

3. **Main Execution Flow**:
   - The code loads two images from disk.
   - It detects and matches features between the two images.
   - It calculates the 3D positions of the matched points.
   - Finally, it visualizes the feature matches and the 3D reconstruction.

## Usage Instructions

### Step 1: Prepare Your Images

- The code expects two images taken from different viewpoints of the same scene. These images should contain overlapping features for accurate keypoint matching.
- Place your images in a directory and replace the paths in the code:

img1 = cv2.imread(r"D:\Desktop\1.jpg")
img2 = cv2.imread(r"D:\Desktop\2.jpg")


Replace `D:\Desktop\1.jpg` and `D:\Desktop\2.jpg` with the paths to your image files.

### Step 2: Run the Script

Once the images are set up, you can run the script in your Python environment:

python 3d_reconstruction.py


### Step 3: View the Results

- The first visualization will show the matched keypoints between the two images, with lines connecting corresponding keypoints.
- The second visualization will display the 3D reconstruction as a scatter plot of the triangulated points.

## Detailed Explanation

### 1. **Feature Detection and Matching with SIFT**

- **SIFT** is used to detect keypoints in the images and compute descriptors for these keypoints. The descriptors are invariant to scale, rotation, and partial affine transformations, making SIFT ideal for matching features between images taken from different viewpoints.
- **FLANN-based matching** is used to find the best matches between the descriptors. FLANN is an efficient way to match high-dimensional data like feature descriptors.

### 2. **Camera Matrix and Distortion Model**

- In this code, a simplified camera matrix (`K`) is used. The matrix contains intrinsic parameters such as focal length and the principal point. The distortion coefficients are assumed to be zero for simplicity.
- In real applications, you should calibrate your camera to obtain accurate intrinsic parameters and distortion coefficients. You can use OpenCV’s camera calibration functions for this.

focal_length = 800  # Example focal length in pixels
center = (640, 360)  # Principal point (image center)
K = np.array([[focal_length, 0, center[0]],
              [0, focal_length, center[1]],
              [0, 0, 1]])


### 3. **Triangulation**

- After matching keypoints between the two images, the next step is to calculate their 3D coordinates.
- The function `cv2.triangulatePoints()` performs the triangulation using the matched keypoints and the projection matrices of the two cameras.
- The projection matrices are formed by combining the intrinsic camera matrix `K` with the relative camera transformation (rotation `R` and translation `T`).
- In this example, a simple identity matrix for rotation and a translation vector is used for illustrative purposes.

### 4. **3D Visualization**

- The triangulated 3D points are normalized to remove the homogeneous coordinate and plotted using `matplotlib`.
- The 3D plot shows the positions of the points in space. The `scatter` function is used to create a 3D scatter plot.

## Notes

- **Camera Calibration**: The example uses a simplified camera matrix with arbitrary focal length and image center. For accurate results, you should calibrate your camera using a calibration pattern (e.g., a checkerboard).
- **Stereo Calibration**: This example assumes the relative camera motion (rotation and translation) is known. In real-world scenarios, you would need to compute the fundamental matrix from the matched keypoints and use stereo calibration to obtain the relative camera position and orientation.
- **Accuracy of 3D Reconstruction**: The accuracy of the 3D reconstruction depends on several factors, including the quality of feature matching, the relative camera motion, and the camera calibration. If too few matches are found or if the points are too close to the camera, the reconstruction might not be accurate.

## Potential Improvements

1. **Fundamental Matrix Estimation**: Instead of assuming an identity rotation and translation, you can use the **fundamental matrix** to compute the relative pose (rotation and translation) between the two cameras.
   
2. **Camera Calibration**: Replace the dummy camera matrix with real calibration data by performing a stereo calibration procedure.

3. **Error Handling**: Add more robust error handling to deal with cases where the number of good matches is too low, or the triangulation fails due to collinear points.

4. **Optimization**: Use bundle adjustment or other optimization techniques to refine the 3D points and camera poses.

## Example Output

1. **Feature Matches**: A plot showing the keypoints and their matches between the two images. Good matches will be connected by lines.
2. **3D Point Cloud**: A 3D scatter plot of the triangulated points, representing the reconstructed scene.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
