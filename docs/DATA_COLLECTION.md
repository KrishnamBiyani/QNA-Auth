# QNA-Auth: Collecting Data for Training

This guide is for **participants** who run the collection script on their laptop and send the data to the project owner for model training.

---

## What you need

- **Python 3.8+** and **pip**
- **Internet** (for QRNG; optional for camera/mic-only)
- **Webcam** (optional; for camera noise)
- **Microphone** (optional; for ambient noise)
- This project cloned (or the collection script + instructions)

---

## Quick steps (participants)

### 1. Get the project

```bash
git clone <repository-url>
cd QNA-Auth
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*(If you only have QRNG and no camera/mic, you can install a minimal set: `pip install numpy requests` and run with `--sources qrng`.)*

### 3. Run the collection script

From the **project root** (`QNA-Auth/`):

```bash
python scripts/collect_data_for_training.py
```

The script will ask for your **name** (e.g. "Alice Laptop") and then collect noise. By default it uses **QRNG only** (internet). To use camera and microphone too:

```bash
python scripts/collect_data_for_training.py --name "Alice Laptop" --sources qrng,camera,microphone --num-samples 50
```

### 4. Send the output folder

When it finishes, a folder is created, e.g.:

- `qna_auth_collection_Alice_Laptop_20250128_120000/`

**Zip it** and send it to the project owner (email, Google Drive, etc.):

```bash
# Optional: create a zip for easy sharing
python scripts/collect_data_for_training.py --name "Alice Laptop" --zip
```

That creates both the folder and a `.zip` file you can send.

---

## Command-line options

| Option | Description | Default |
|--------|-------------|---------|
| `--name "Your Name"` | Participant/device name | Prompts if omitted |
| `--sources qrng,camera,microphone` | Which sources to use | `qrng` |
| `--num-samples 50` | Samples per source (10–200) | 50 |
| `--output-dir PATH` | Where to save the folder | `./qna_auth_collection_<name>_<timestamp>` |
| `--zip` | Also create a .zip of the folder | Off |

**Examples:**

- QRNG only (no camera/mic):  
  `python scripts/collect_data_for_training.py --name "Bob"`
- Camera + mic (no internet):  
  `python scripts/collect_data_for_training.py --name "Carol" --sources camera,microphone --num-samples 30`
- All sources and create zip:  
  `python scripts/collect_data_for_training.py --name "Dave" --sources qrng,camera,microphone --num-samples 50 --zip`

---

## What gets collected

- **QRNG**: Quantum random numbers from an online API (needs internet).
- **Camera**: Dark-frame sensor noise from the webcam (a few seconds, dim/no cover).
- **Microphone**: Short ambient noise recordings (browser may ask for permission).

Data is saved as raw arrays (`.npy`) plus a `manifest.json` with device name and counts. No personal data is collected beyond the name you give.

---

## Troubleshooting

- **"No module named 'noise_collection'"**  
  Run the script from the **project root** (`QNA-Auth/`), not from inside `scripts/`.

- **QRNG fails**  
  Check internet; ANU API can have rate limits. Retry or use `--sources camera,microphone` only.

- **Camera/Mic not found**  
  Use `--sources qrng` only, or fix camera/mic permissions and try again.

- **Want to skip camera/mic**  
  Use `--sources qrng` so only internet-based QRNG is used.

---

## For the project owner: ingesting received data

After receiving folders (or zips) from participants:

```bash
python scripts/ingest_collected_data.py path/to/folder1 path/to/folder2.zip ...
```

This merges all samples into `dataset/samples/` so you can run the normal training pipeline.
