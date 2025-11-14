from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml


def predict_and_save(model, image_path, output_path, output_path_txt):
    results = model.predict(image_path, conf=0.5)
    result = results[0]

    img = result.plot()

    cv2.imwrite(str(output_path), img)

    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':

    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        images_dir = Path(data['test']) / 'images'

    if not images_dir.exists():
        print(f"Test folder images not found: {images_dir}")
        exit()

    detect_path = this_dir / "runs" / "detect"
    train_folders = sorted([
        f for f in os.listdir(detect_path)
        if os.path.isdir(detect_path / f) and f.startswith("train")
    ])

    latest_folder = train_folders[-1]
    model_path = detect_path / latest_folder / "weights" / "best.pt"

    print(f"Using model: {model_path}")

    model = YOLO(model_path)

    output_dir = this_dir / "predictions"
    images_output_dir = output_dir / "images"
    labels_output_dir = output_dir / "labels"

    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images_dir.glob('*'):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        output_img = images_output_dir / img_path.name
        output_txt = labels_output_dir / (img_path.stem + '.txt')

        predict_and_save(model, img_path, output_img, output_txt)

    print(f"Predictions complete ✔")
    print(f"Images saved at: {images_output_dir}")
    print(f"Labels saved at: {labels_output_dir}")

    # full test evaluation
    print("\nRunning test evaluation...")
    metrics = model.val(data=this_dir / 'yolo_params.yaml', split="test")
    print("Evaluation complete!")
