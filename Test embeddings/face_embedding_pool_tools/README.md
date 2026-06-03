# Face Embedding Pool Tools for the OpenCV/Moodle Station

These tools generate embedding JSON files that your station can load directly.

Your station accepts this generated format:

```json
{
  "name": "Person Name",
  "model": {"family": "opencv", "detector": "yunet", "recognizer": "sface"},
  "embeddings": [[0.01, -0.02, "... 128 floats ..."]]
}
```

## 1. Install requirements

Run inside your existing station folder/venv:

```powershell
cd C:\Users\Ghinescu Lucian\Desktop\Master\DISERTATIE\Cod\Face Recognition ex\camera_station\opencv_station
py -m pip install opencv-contrib-python numpy tqdm
```

If you already have OpenCV working in the station, you may only need:

```powershell
py -m pip install numpy tqdm
```

## 2. Generate real SFace embeddings from LFW

Copy `generate_sface_embeddings.py` into your `opencv_station` folder, then run:

```powershell
py .\generate_sface_embeddings.py --download-lfw --output .\embedding_pool_lfw_200x5 --max-people 200 --images-per-person 5 --min-images-per-person 2 --shuffle
```

That creates roughly:

```text
200 people × up to 5 embeddings = up to 1000 embeddings
```

Use this output folder in your desktop app's embeddings selector:

```text
embedding_pool_lfw_200x5
```

## 3. Bigger test

```powershell
py .\generate_sface_embeddings.py --download-lfw --output .\embedding_pool_lfw_1000x5 --max-people 1000 --images-per-person 5 --min-images-per-person 2 --shuffle
```

## 4. Generate embeddings from your own folder

Folder structure:

```text
my_dataset/
  Student_001/
    1.jpg
    2.jpg
  Student_002/
    1.jpg
    2.jpg
```

Command:

```powershell
py .\generate_sface_embeddings.py --input-folder .\my_dataset --output .\embedding_pool_my_dataset --max-people 0 --images-per-person 10
```

## 5. Synthetic stress pool for pure performance tests

This does not test recognition accuracy. It only tests how your station behaves when many vectors are loaded.

```powershell
py .\make_synthetic_stress_pool.py --people 5000 --embeddings-per-person 5 --output .\synthetic_25k_embeddings
```

Then select `synthetic_25k_embeddings` in your station. Your real face will probably become `Unknown`; that is expected because the vectors are random.

## 6. Recommended experiment sequence

```text
A: 200 people × 5 embeddings  = 1,000 embeddings
B: 500 people × 5 embeddings  = 2,500 embeddings
C: 1000 people × 5 embeddings = 5,000 embeddings
D: synthetic 5000 × 5         = 25,000 vectors, performance-only
```

For real attendance safety, test thresholds:

```text
0.36, 0.40, 0.45
```

Wrong attendance markings are worse than Unknown detections.
