import cv2 
from pathlib import Path
import pickle
import numpy as np


def edge_smoothing(image, mask, ksize=(5, 5), sigma=3, thickness=5):
    """
    Smooth out edges of segmented raspberries by applying gaussian filtering
    only on the contour area, then overlay back on the original image.
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find contours and draw them as a thick band
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=thickness)

    # Blur the original image
    blurred = cv2.GaussianBlur(image, ksize, sigma)

    # Swap in blurred pixels only where the contour is
    result = image.copy()
    result[contour_mask > 0] = blurred[contour_mask > 0]
    # TODO : possibly also need to adjust mask ???

    cv2.imwrite("edge_smoothed_image.png", result)
    

    return result, mask



def normalize_distribution(image,mask,target_mean = 128, target_std = 40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    roi = v[mask > 0]
   # target_mean = target_mean.mean()
   # target_std = target_std.mean()
    v[mask > 0] = np.clip((roi - roi.mean()) / (roi.std() + 1e-6) * target_std + target_mean, 0, 255).astype(np.uint8)
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    cv2.imwrite("normalized_image.png", result)

    return result, mask



def find_white_pellets():
    pass 


def overlay_mask_on_image(image, mask):
    # Create a red mask
    red_mask = np.zeros_like(image)
    green_mask = np.zeros_like(image)
    green_mask[mask > 0] = [0, 255, 0]  # Green color

    # Blend the original image with the red mask
    blended = cv2.addWeighted(image, 0.8, green_mask, 0.2, 0)

    cv2.imwrite("overlayed_image.png", blended)

def main():
    # Load the dataset (anomalous and non-anomalous samples)
    dataset_path = Path("../../disk/dataset_single_objects/GT/processed")
    normal_path = dataset_path / 'normal' / 'normal_samples.pkl'
    anomalous_path = dataset_path / 'anomalous' / 'anomalous_samples.pkl'

    # Get examplary normal and anomalous sample
    with open(normal_path, 'rb') as f:
        normal_data = pickle.load(f)
    with open(anomalous_path, 'rb') as f:
        anomalous_data = pickle.load(f)

    # Concatenate normal and anomalous data
    all_images = [item['image'] for item in normal_data + anomalous_data]
    all_masks = [item['mask'] for item in normal_data + anomalous_data]

    # Convert all images to hsv and only look at the v channel for mean and std calculation
    all_hsv_images = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in all_images]
    all_v_channels = [hsv[:,:,2] for hsv in all_hsv_images]

    all_pixels = np.concatenate([v[mask > 0] for v, mask in zip(all_v_channels, all_masks)])
    target_mean = all_pixels.mean(axis=0)
    target_std = all_pixels.std(axis=0) 

    print(f"Calculated target mean: {target_mean}, target std: {target_std}")

    
   # normal_sample = normal_data[0]['image']
   # anomalous_sample = anomalous_data[0]['image']
    print(len(normal_data), len(anomalous_data))
    
    for i in range(len(normal_data)):
        if i == 0:
            normal_sample = normal_data[i]['image']
            normal_mask = normal_data[i]['mask']

            # Turn to RGB for visualization
            normal_sample = cv2.cvtColor(normal_sample, cv2.COLOR_BGR2RGB)
            cv2.imwrite("original_normal_image.png", normal_sample)

          #  edge_smoothing(normal_sample, normal_mask)
          #  overlay_mask_on_image(normal_sample, normal_mask)
            normalize_distribution(normal_sample, normal_mask, target_mean, target_std)
            break
    







if __name__ == "__main__":
    main()
