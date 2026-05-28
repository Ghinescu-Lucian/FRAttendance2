# Facial recognition model test variants

This package contains four separate variants for testing different facial-recognition models with the same idea:

1. Load known people from a folder of subject images.
2. Extract one embedding per valid subject image.
3. Open the camera.
4. Detect all visible faces.
5. Compare each detected face against the gallery.
6. Draw green boxes for recognized people and red boxes for unknown people.

## Recommended folder layout

Use one folder per person:

```text
images/
  GhinescuLucian/
    front.jpg
    left.jpg
    right.jpg
  IuliaSocarde/
    front.jpg
    left.jpg
```

Flat files also work, but folders are safer:

```text
images/
  GhinescuLucian_1.jpg
  GhinescuLucian_2.jpg
  IuliaSocarde_1.jpg
```

## Variant 01 — OpenCV YuNet + SFace

Closest to your current program.

Install:

```powershell
py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
py -m pip install opencv-contrib-python numpy
```

Run:

```powershell
py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images
```

Useful options:

```powershell
py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images --match-threshold 0.36 --det-threshold 0.70
```

## Variant 02 — InsightFace buffalo_l / buffalo_s

This is the most important one to test for your dissertation because it usually gives much stronger embeddings than SFace, especially for difficult classroom conditions.

CPU install:

```powershell
py -m pip install insightface onnxruntime opencv-python numpy
```

GPU install, only after your CUDA environment is correct:

```powershell
py -m pip install insightface onnxruntime-gpu opencv-python numpy
```

Run on CPU:

```powershell
py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_l
```

Run on GPU:

```powershell
py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_l --gpu
```

Try smaller/faster model:

```powershell
py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_s
```

## Variant 03 — dlib / face_recognition

Classic 128D baseline. Easy to compare, but normally not the best choice for classroom-distance recognition.

Install:

```powershell
py -m pip install face_recognition opencv-python numpy
```

Run:

```powershell
py variant_03_face_recognition_dlib.py --subjects-dir images
```

If it is too slow:

```powershell
py variant_03_face_recognition_dlib.py --subjects-dir images --resize 0.35
```

## Variant 04 — DeepFace

Convenient wrapper for testing several models, but slower for real-time camera work.

Install:

```powershell
py -m pip install deepface tensorflow opencv-python numpy
```

Run default Facenet512:

```powershell
py variant_04_deepface_facenet512.py --subjects-dir images
```

Try ArcFace:

```powershell
py variant_04_deepface_facenet512.py --subjects-dir images --model-name ArcFace
```

Try faster detector:

```powershell
py variant_04_deepface_facenet512.py --subjects-dir images --detector-backend opencv
```

## Threshold tuning

Thresholds are not universal. Start with these defaults:

| Variant | Metric | Default threshold | Meaning |
|---|---:|---:|---|
| OpenCV SFace | cosine similarity | 0.36 | Higher = stricter |
| InsightFace | cosine similarity | 0.35 | Higher = stricter |
| dlib/face_recognition | Euclidean distance | 0.55 | Lower = stricter |
| DeepFace | cosine similarity | 0.45 | Higher = stricter |

For attendance, you should tune thresholds using real images from your classroom camera, not clean close-up phone photos only.

## Practical recommendation

Test in this order:

1. `variant_02_insightface_buffalo.py`
2. `variant_01_opencv_yunet_sface_from_images.py`
3. `variant_04_deepface_facenet512.py`
4. `variant_03_face_recognition_dlib.py`

For your current project, InsightFace is the best candidate to compare against YuNet/SFace because it is still script-friendly, supports GPU acceleration through ONNX Runtime, and produces strong face embeddings.
