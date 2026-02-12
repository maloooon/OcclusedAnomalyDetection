# Our aim is to create a much smaller model for segmentation, specific for our task of segmenting raspberries, such that we can have fast inference times.
# The initial idea is Model Distillation , but since this can add quite some complexity, we first try with an easier approach : 
# Train a smaller model on pseudo-labels (in this case segmentation masks) provided by a large model (e.g. SAM3) and then evaluate on the GT test set :
# See how good performance is 


from datasets import load_dataset
import numpy as np 
from pathlib import Path
import pickle
import shutil
from ultralytics import YOLO


def setup_yolo_dataset_structure_extended(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES, N_VAL_SAMPLES):

    images_dir_train = Path("../../disk/YOLO_dataset_extended/images/train")
    images_dir_val   = Path("../../disk/YOLO_dataset_extended/images/val")
    images_dir_test  = Path("../../disk/YOLO_dataset_extended/images/test")

    for d in [images_dir_train, images_dir_val, images_dir_test]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for idx, (img, img_name) in enumerate(zip(all_imgs, all_pred_ids)):
        img_name = f"{img_name}.png"
        if idx < N_TRAIN_SAMPLES:
            img_path = images_dir_train / img_name
        elif idx < N_TRAIN_SAMPLES + N_VAL_SAMPLES:
            img_path = images_dir_val / img_name
        else:
            img_path = images_dir_test / img_name

        img.save(img_path)

    img_paths = (
        sorted(list(images_dir_train.glob("*.png"))) +
        sorted(list(images_dir_val.glob("*.png")))   +
        sorted(list(images_dir_test.glob("*.png")))
    )

    polygons_per_image = all_pred_xyn

    # Replace val AND test labels with GT labels (not pseudo labels)
    val_labels = [[np.array(label) for label in sample_labels] for sample_labels in val_labels]
    polygons_per_image = polygons_per_image[:N_TRAIN_SAMPLES] + val_labels

    labels_dir_train = Path("../../disk/YOLO_dataset_extended/labels/train")
    labels_dir_val   = Path("../../disk/YOLO_dataset_extended/labels/val")
    labels_dir_test  = Path("../../disk/YOLO_dataset_extended/labels/test")

    for d in [labels_dir_train, labels_dir_val, labels_dir_test]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, polys) in enumerate(zip(img_paths, polygons_per_image)):

        img_name = Path(img_path).stem
        if idx < N_TRAIN_SAMPLES:
            label_file = labels_dir_train / f"{img_name}.txt"
        elif idx < N_TRAIN_SAMPLES + N_VAL_SAMPLES:
            label_file = labels_dir_val / f"{img_name}.txt"
        else:
            label_file = labels_dir_test / f"{img_name}.txt"

        if label_file.exists():
            label_file.unlink()

        lines = []
        for poly in polys:
            coords = poly.flatten()
            line = "0 " + " ".join(map(str, coords))
            lines.append(line)

        label_file.write_text("\n".join(lines))

def setup_yolo_dataset_structure(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES):

    # Adjust to YOLO format : https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format

    # First off, YOLO format requires images to be in a folder structure, with labels in a separate folder.
    # So let us first deal with the images 
    

    images_dir_train = Path("../../disk/YOLO_dataset/images/train")
    images_dir_val = Path("../../disk/YOLO_dataset/images/val")

    # Remove directories if they exist
    if images_dir_train.exists():
        shutil.rmtree(images_dir_train)
    if images_dir_val.exists():
        shutil.rmtree(images_dir_val)

    # Recreate directories
    images_dir_train.mkdir(parents=True, exist_ok=True)
    images_dir_val.mkdir(parents=True, exist_ok=True)

    for idx, (img, img_name) in enumerate(zip(all_imgs, all_pred_ids)):
        img_name = f"{img_name}.png"
        if idx < N_TRAIN_SAMPLES:
            img_path = images_dir_train / img_name
        else:
            img_path = images_dir_val / img_name

        img.save(img_path)

    img_paths = sorted(list(images_dir_train.glob("*.png"))) + sorted(list(images_dir_val.glob("*.png")))
    polygons_per_image = all_pred_xyn

    # Replace the above with the GT labels for the val set, since we want to evaluate on the GT labels, not on the predicted masks
    # val_labels is a list of lists of polygons, where the outer list is over images and the inner list is over objects in the image
    # Turn into the same format as polygons_per_image, which is a list of lists of numpy arrays

    val_labels = [[np.array(label) for label in sample_labels] for sample_labels in val_labels]

    # Now replace
    polygons_per_image = polygons_per_image[:N_TRAIN_SAMPLES] + val_labels

    labels_dir_train = Path("../../disk/YOLO_dataset/labels/train") 
    labels_dir_val = Path("../../disk/YOLO_dataset/labels/val")

    if labels_dir_train.exists():
        shutil.rmtree(labels_dir_train)
    if labels_dir_val.exists():
        shutil.rmtree(labels_dir_val)

    labels_dir_train.mkdir(parents=True, exist_ok=True)
    labels_dir_val.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, polys) in enumerate(zip(img_paths, polygons_per_image)):

        img_name = Path(img_path).stem
        if idx < N_TRAIN_SAMPLES:
            label_file = labels_dir_train / f"{img_name}.txt"
        else:
            label_file = labels_dir_val / f"{img_name}.txt"


        # Check if label file already exists, if so, delete and create a new one (to avoid appending to existing file)
        if label_file.exists():
            label_file.unlink()


        lines = []


        for poly in polys:


            # flatten x,y pairs to get it into wanted YOLO shape : x1 y1 x2 y2 x3 y3 ... xn yn
            coords = poly.flatten()
 


            # NOTE : for now, just class index 0 since all objects are raspberries ; need to see if we need to adjust
            # NOTE: for raspberry grades or not
            # Creates class_index x1 y1 x2 y2 ... xn yn
            line = "0 " + " ".join(map(str, coords))
            lines.append(line)

        label_file.write_text("\n".join(lines))





def main():
    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")


    # Define the given data split : 160 training, 40 validation samples
    # Note that we will take the N_TRAIN_SAMPLES from our large model (e.g. SAM3),
    # but the N_VAL_SAMPLES from the original GT.
    N_TRAIN_SAMPLES = 160
    N_VAL_SAMPLES = 20
    N_TEST_SAMPLES = 20

    # Load GT masks for evaluation later on
    val_data = list(ds['valid'])
    
    
    val_labels = [sample['labels'] for sample in val_data]

    # Remove bonnet mask (class 0)
    val_labels = [[label for label in sample_labels if label[0] != 0] for sample_labels in val_labels]

    # Do not record the raspberry grade, since we are only interested in segmentation performance for now, not classification performance
    val_labels = [[label[1:] for label in sample_labels] for sample_labels in val_labels]

    ## Load predicted masks
    PRED_MASKS_FILE = '../../disk/saved_masks/SAM3/masks.pkl'
    with open(PRED_MASKS_FILE, 'rb') as f:
        pred_data = pickle.load(f)

    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]


    # Sort by image ID to ensure alignment
    all_pred_masks_ids.sort(key=lambda x: x[1])

    all_pred_ids = [img_id for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_pred_masks = [masks_and_xyn_and_imgs[0] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_pred_xyn = [masks_and_xyn_and_imgs[1] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_imgs = [masks_and_xyn_and_imgs[2] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]


   # setup_yolo_dataset_structure(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES)
 #   setup_yolo_dataset_structure_extended(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES, N_VAL_SAMPLES)



    # Load a model
  #  model = YOLO("../../disk/pretrained_models/yolo26n-seg.pt")  # Load pretrained model
    # Train the model
  #  results = model.train(data="../../disk/YOLO_dataset_extended/RaspGrade-seg.yaml", epochs=100, imgsz=640, name = "yolo26n-seg-pseudo-labels-extended", device = 2)

    # Load trained model
    model = YOLO("/home/marlon_helbing/MasterThesis/OcclusedAnomalyDetection/runs/segment/yolo26n-seg-pseudo-labels/weights/best.pt")
   # metrics = model.val(data="../../disk/YOLO_dataset/RaspGrade-seg.yaml", device = 'cpu')
   # print(metrics.seg.f1)


    # Get inference time by evaluating on 5 random samples in /home/marlon_helbing/disk/YOLO_dataset/images/val

    import time 
    val_img_paths = sorted(list(Path("../../disk/YOLO_dataset/images/val").glob("*.png")))
    sample_paths = np.random.choice(val_img_paths, size=5, replace=False)
    
    for i, img_path in enumerate(sample_paths):
        if i == 1:  # Skip the first one to avoid including any potential model loading time
            start_time = time.time()
        model.predict(img_path, device = 2)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / (len(sample_paths) -1)
    print(f"Average inference time on GPU for 5 samples: {avg_inference_time:.4f} seconds")
         













if __name__ == "__main__":
    main()
    
