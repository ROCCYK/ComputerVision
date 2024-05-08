import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def align_images(img, ref_img):
    # Convert images to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_ref_img, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp image
    height, width, channels = ref_img.shape
    aligned_img = cv2.warpPerspective(img, h, (width, height))

    return aligned_img

def abs_diff_threshold(img1, img2):
    # Compute the absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Threshold the difference
    _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    return thresh

def compute_ssim(img1, img2):
    # Compute SSIM
    _, diff = ssim(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

def feature_matching(img1, img2):
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect and compute keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher and match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw matches
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    return matched_img

if __name__ == "__main__":
    # Bannana
    # Load the two images
    image1 = cv2.imread('images/banana.jpg')
    image2 = cv2.imread('images/banana2.jpg')

    # Ensure both images are of the same size for direct comparison methods
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    # applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")
    
    # Bridge
    # Load the two images
    image1 = cv2.imread('images/bridge1.jpg')
    image2 = cv2.imread('images/bridge2.jpg')
    
    # applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")
    
    # City
    # Load the two images
    image1 = cv2.imread('images/city1.jpg')
    image2 = cv2.imread('images/city2.jpg')
    
    # applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")
    
    # Fish
    # Load the two images
    image1 = cv2.imread('images/clownfish_1.jpg')
    image2 = cv2.imread('images/clownfish_2.jpg')
    
    # applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")
    
    # Desert
    # Load the two images
    image1 = cv2.imread('images/desert1.jpg')
    image2 = cv2.imread('images/desert2.jpg')
    
    # applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")
    
    # Snow
    # Load the two images
    image1 = cv2.imread('images/snow1.png')
    image2 = cv2.imread('images/snow2.png')
    
    # Align image 2 with image 1
    image2 = align_images(image2,image1)
    
    # Applying GaussianBlur to reduce noise
    image1 = cv2.GaussianBlur(image1, (9, 9), 0)
    image2 = cv2.GaussianBlur(image2, (9, 9), 0)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    thresh = abs_diff_threshold(gray1, gray2)
    show_image(image=thresh, title="Absolute Difference Threshold")
    
    ssim_thresh = compute_ssim(gray1, gray2)
    show_image(image=ssim_thresh, title="SSIM Differences")
    
    matched_img = feature_matching(gray1, gray2)
    show_image(image=matched_img, title="Feature Matching")