from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont




def visualize_sample(sample, show_segmentation_mask=False, show_classes=False, debug=False):
    """
    Visualize a sample from the training set.
    
    Args:
        sample: Dictionary containing data
        show_segmentation_mask: Whether to overlay segmentation masks
        show_classes: Whether to show class labels at mask centers
        debug: Print debug information about label structure
    """
    img = sample['image'] # Get PIL image
    labels = sample['labels'] # Get label and segmentation mask points
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(img)
    width, height = img.size
    
    print(f"Number of raspberries and bonnet (box): {len(labels)}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.axis('off')
    
    if show_segmentation_mask or show_classes:
        # Create a mask overlay
        mask_overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(mask_overlay)
        
        font = ImageFont.load_default()
        
        colors = [
            (200, 200, 200, 80),   # Class 0: Gray (punnet)
            (139, 69, 19, 120),     # Class 1: Brown
            (220, 20, 60, 120),     # Class 2: Red
            (255, 182, 193, 120),   # Class 3: Pink
            (154, 205, 50, 120),    # Class 4: Yellow-green
            (128, 0, 128, 120)      # Class 5: Purple 
        ]

        # Sort labels: draw class 0 (box) first, then others
        # This ensures the box doesn't overlay the raspberries
        sorted_labels = sorted(enumerate(labels), key=lambda x: (0 if int(x[1][0]) == 0 else 1, x[0]))
        
        drawn_count = 0
        for original_idx, label_data in sorted_labels:
            
            if debug:
                print(f"\nSample {original_idx}:")
                print(f"  Total values: {len(label_data)}")
                print(f"  First 5 values: {label_data[:5]}")
                print(f"  Class ID: {int(label_data[0])}")
            
            if len(label_data) < 3:  # Need at least class + one point (x, y)
                if debug:
                    print(f"  SKIPPED: Not enough data")
                continue
                
            class_id = int(label_data[0])
            
            # Extract polygon coordinates (normalized values)
            coords = label_data[1:]
            
            # Convert normalized coordinates to pixel coordinates
            polygon_points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * width
                    y = coords[i + 1] * height
                    polygon_points.append((x, y))
            
            if len(polygon_points) < 3:  # Need at least 3 points for a polygon
                if debug:
                    print(f"  SKIPPED: Only {len(polygon_points)} points")
                continue

            if debug:
                print(f"  Polygon points: {len(polygon_points)}")
            
            # Draw segmentation mask
            if show_segmentation_mask:
                color = colors[class_id % len(colors)]
                try:
                    draw.polygon(polygon_points, fill=color, outline=color[:3] + (255,))
                    if debug:
                        print(f"  Drew segmentation mask with color {color}")
                except Exception as e:
                    print(f"  ERROR drawing polygon: {e}")
                    continue
            
            # Draw class label at centroid
            if show_classes:
                if class_id > 0: # do not draw class label for the punnet
                    try:
                        # Calculate centroid
                        centroid_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                        centroid_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                        
                        # Draw class label
                        text = str(class_id)
                        bbox = draw.textbbox((centroid_x, centroid_y), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Draw background rectangle for better visibility
                        padding = 5
                        draw.rectangle(
                            [(centroid_x - text_width/2 - padding, centroid_y - text_height/2 - padding),
                            (centroid_x + text_width/2 + padding, centroid_y + text_height/2 + padding)],
                            fill=(255, 255, 255, 200)
                        )
                        
                        # Draw text
                        draw.text(
                            (centroid_x - text_width/2, centroid_y - text_height/2),
                            text,
                            fill=(0, 0, 0, 255),
                            font=font
                        )
                        if debug:
                            print(f"  Drew class label '{text}' at ({centroid_x:.1f}, {centroid_y:.1f})")
                    except Exception as e:
                        print(f"  ERROR drawing class label: {e}")
                        continue
            
            drawn_count += 1
        
        print(f"\nTotal drawn: {drawn_count}/{len(labels)}")
        
        # Overlay the mask on the image
        if show_segmentation_mask or show_classes:
            img_with_overlay = Image.alpha_composite(img.convert('RGBA'), mask_overlay)
            ax.imshow(np.array(img_with_overlay))
    
    plt.tight_layout()
    plt.show()

def distribution_of_grades(samples):
    """
    Plot the distribution of grades in the dataset.
    
    Args:
        samples: List of samples from the dataset
    """

    # Count grades for plotting
    grade_counts = {}
    
    for sample in samples:
        labels = sample['labels']
        for label_data in labels:
            class_id = int(label_data[0])
            if class_id > 0:  # Exclude class 0 (punnet)
                if class_id not in grade_counts:
                    grade_counts[class_id] = 0
                grade_counts[class_id] += 1
    
    # Prepare data for plotting
    classes = sorted(grade_counts.keys())
    counts = [grade_counts[c] for c in classes]
    
    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Distribution of Grades in Dataset')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    


def main():
    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

    # Visualize a random sample from the training set
    sample = ds['train'][0]
    visualize_sample(sample, show_segmentation_mask=True, show_classes=True, debug=True)

    # Plot distribution of grades
    #distribution_of_grades(list(ds['train']) + list(ds['valid']))


if __name__ == "__main__":
    main()