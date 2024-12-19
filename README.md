# YOLO Confidence Enhancement with SVM

This project aims to enhance the performance of the YOLOv11 object detection model by modifying input image properties (e.g., saturation, hue, brightness) based on confidence and IoU scores. The approach leverages Support Vector Machine (SVM) to optimize image adjustments, improving the overall model scores.

## Features

- Utilizes YOLOv11 for object detection.
- Extracts confidence and IoU scores for analysis.
- Implements an SVM-based model to optimize image parameters such as:
  - Saturation
  - Hue
  - Brightness
  - Contrast
- Dynamically adjusts image properties to achieve better detection scores.

## Setup

### Prerequisites

- Python 3.8 or later
- CUDA-compatible GPU (recommended)
- Installed dependencies (e.g., `ultralytics`, `torch`, `scikit-learn`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yolo-confidence-enhancement.git
   cd yolo-confidence-enhancement
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify CUDA setup for faster computations:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

## Usage

### Training the SVM Model

1. **Prepare Dataset**: Ensure you have a labeled dataset for YOLOv11 object detection.
2. **Run YOLO Inference**: Use YOLOv11 to extract confidence and IoU scores for the images in the dataset.
3. **Train SVM**: Train the SVM model using the extracted scores and image properties to learn the optimal modifications.

   ```python
   from sklearn.svm import SVR
   svr = SVR(kernel='rbf')
   svr.fit(X_train, y_train)  # X_train: image properties, y_train: model scores
   ```

### Image Modification

- Apply transformations such as saturation, hue, brightness, and contrast using libraries like OpenCV or PIL:

   ```python
   from PIL import Image, ImageEnhance

   image = Image.open("image.jpg")
   enhancer = ImageEnhance.Brightness(image)
   enhanced_image = enhancer.enhance(1.5)  # Increase brightness by 50%
   ```

### Evaluation

1. Test the enhanced images with YOLOv11 and observe improved confidence and IoU scores.
2. Analyze the results using metrics such as mAP (mean Average Precision).

## Future Work

- Extend the model to other YOLO versions (e.g., YOLOv5, YOLOv8).
- Explore other machine learning techniques for image optimization.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any feature suggestions or bug fixes.


## Acknowledgments

- [YOLO by Ultralytics](https://github.com/ultralytics/yolov5)
- Scikit-learn for SVM implementation.
- OpenCV and PIL for image processing.

