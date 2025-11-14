EPOCHS = 10
MOSAIC = 0.4
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.0001
LRF = 0.0001
SINGLE_CLS = False

import argparse
from ultralytics import YOLO
import os

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load YOLOv8s model
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))

    # ✅ CPU-friendly training setup for Mac
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=8,                   # keep small batch for CPU stability
        device='cpu',              # important fix for MacBooks
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        workers=2,                 # reduce CPU threads usage
        verbose=True               # show detailed training logs
    )

    print("\n✅ Training complete! Check the 'runs/detect/train' folder for results.\n")
