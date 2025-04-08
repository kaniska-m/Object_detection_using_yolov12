# 🍅 Object Detection Using YOLOv12-Nano on Raspberry Pi

This repository showcases a **lightweight, real-time object detection system** using the **YOLOv12-Nano** model, trained for tomato ripeness classification and deployed on a **Raspberry Pi**. The project demonstrates efficient AI inference on edge devices for smart agriculture applications.

---

## 🚀 Features

- Real-time object detection using YOLOv12-Nano.
- Optimized model deployment using **TFLite** for Raspberry Pi.
- Custom dataset with **augmented training** for robust detection.
- Python-based **Flask web UI** for live streaming detection output.
- Low-latency performance for edge computing applications.

---

## 📂 Repository Structure

```
├── model/
│   ├── yolov12-nano.tflite          # TFLite optimized model
│   └── labelmap.txt                 # Class labels
├── dataset/
│   └── ...                          # Image dataset (optional link to external source)
├── scripts/
│   ├── detect.py                    # Real-time detection script
│   └── utils.py                     # Helper functions
├── web_ui/
│   ├── app.py                       # Flask-based UI for live streaming
│   └── templates/
│       └── index.html               # Web interface
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

Install the required dependencies on Raspberry Pi (Python 3.7+ recommended):

```bash
pip install -r requirements.txt
```

Additional dependencies:
- `opencv-python`
- `tensorflow==2.10.0` *(use lite version for Pi)*
- `flask`
- `imutils`

---

## 📸 Dataset and Preprocessing

- 482 real-world tomato images collected and annotated (YOLO format).
- Augmentation applied:
  - Rotation
  - Brightness variation
  - Gaussian blur
- Final dataset size: **1930 images**
- Image resolution: **640 × 640**
- Labels: `["unripe", "semi-ripe", "ripe"]`

[Click here to view dataset (optional link)](https://example.com)

---

## 🧠 Model Training

Training was performed on Google Colab with GPU acceleration using the **YOLOv12-Nano** architecture.

### Training Script (Colab):

```python
!python train.py \
  --data dataset.yaml \
  --cfg yolov12-nano.yaml \
  --weights '' \
  --epochs 100 \
  --img 640
```

Post-training, the model was **quantized** and exported to **TensorFlow Lite (TFLite)** using:

```python
converter = tf.lite.TFLiteConverter.from_saved_model('best_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

---

## 🧪 Live Detection on Raspberry Pi

### Run the detection script:

```bash
python3 scripts/detect.py --model model/yolov12-nano.tflite --labels model/labelmap.txt --camera 0
```

### Optional: Run with Flask UI:

```bash
cd web_ui
python3 app.py
```

Open your browser at `http://<raspberry_pi_ip>:5000` to view live detection.

---

## 📊 Performance

| Metric       | Value                    |
|--------------|---------------------------|
| FPS (avg)    | ~10 FPS (Raspberry Pi 4)  |
| Model Size   | < 5 MB (TFLite)           |
| Accuracy     | 91.2% mAP@0.5             |

---

## 📌 Applications

- Smart agricultural harvesting
- Tomato ripeness sorting
- Autonomous robotic arms for crop selection

---

## 🙌 Acknowledgments

- [YOLOv12 authors](https://github.com/YOLOv12)
- [TensorFlow Lite](https://www.tensorflow.org/lite/)
- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)

---

## 📃 License

This project is licensed under the MIT License. See `LICENSE` for more information.
