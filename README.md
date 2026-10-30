# Dental X-ray frozen-feature benchmark

This benchmark compares four pretrained encoder families under the same
small-data classification protocol:

- ResNet18
- DINOv3 ViT-S/16
- SigLIP2 Base Patch16
- MambaVision-T

All backbones are frozen. Each image is embedded once, then every fold uses the
same `StandardScaler -> PCA(64) -> class-balanced LogisticRegression` pipeline.

## Dataset layout

For a binary Healthy/Caries experiment:

```text
dataset/
├── Healthy Teeth/
│   ├── image_001.jpg
│   └── image_002.jpg
└── Caries/
    ├── image_101.jpg
    └── image_102.jpg
```

Nested directories below each class are supported.

If augmented images are present, do not randomly split them independently from
their originals. Instead, create a manifest with an `original_image_id` or
`patient_id` column and run the Python script with `--group-column`.

## Setup

```bash
chmod +x setup_env.sh run_experiment.sh
./setup_env.sh
```

For Linux, Python 3.11, and CUDA 12.6:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
PYTHON_BIN=python3.11 \
./setup_env.sh
```

Install the optional Mamba CUDA dependency at setup time with:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
PYTHON_BIN=python3.11 \
INSTALL_MAMBA=1 \
./setup_env.sh
```

If the base environment already exists, add Mamba support without installing
the `mambavision` PyPI package:

```bash
source .venv/bin/activate
pip uninstall -y mambavision mamba-ssm
pip install packaging ninja timm==1.0.15 einops==0.8.1
pip install mamba-ssm==2.3.2.post1 --no-build-isolation
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba-ssm OK')"
```

The Hugging Face MambaVision implementation imports `mamba_ssm` directly; the
separate `mambavision` package is not required by this benchmark.

The default run uses the three dependency-light backbones. After the optional
Mamba installation succeeds, include it explicitly:

```bash
MODELS="resnet18 dinov3_vits16 siglip2 mambavision_t" ./run_binary.sh
```

DINOv3 is a gated model. Visit
`facebook/dinov3-vits16-pretrain-lvd1689m` on Hugging Face, accept its license,
then authenticate:

```bash
source .venv/bin/activate
hf auth login
```

## Run all models

```bash
./run_experiment.sh /absolute/path/to/dataset
```

For the primary Healthy-vs-Caries experiment, the repository includes a
dedicated runner. If the dataset is in the default location shown below, no
argument is required:

```bash
./run_binary.sh
```

Otherwise pass the classification directory:

```bash
./run_binary.sh "/absolute/path/to/Dental OPG (Classification)"
```

The optional six-class experiment is run separately:

```bash
./run_sixclass.sh "/absolute/path/to/Dental OPG (Classification)"
```

For the Dental OPG dataset shown in the example project, use its classification
directory as the data root:

```bash
./run_experiment.sh "data/Dental OPG XRAY Dataset/Dental OPG (Classification)"
```

The shell runner defaults to the two classes `Healthy Teeth` and `Caries`.
Specify other class names as a comma-separated list so spaces are preserved:

```bash
CLASSES="Healthy Teeth,Caries,Impacted Teeth" \
./run_experiment.sh "data/Dental OPG XRAY Dataset/Dental OPG (Classification)"
```

Outputs are written to `outputs/`:

```text
outputs/
├── splits.csv
├── run_config.json
├── fold_metrics.csv
├── predictions.csv
├── summary.csv
└── features/
    ├── resnet18.npz
    ├── dinov3_vits16.npz
    ├── siglip2.npz
    └── mambavision_t.npz
```

Embedding files are cached. Re-running the script reuses them when the dataset
has not changed.

## Run a smaller subset first

ResNet18 and MambaVision-T:

```bash
MODELS="resnet18 mambavision_t" ./run_experiment.sh /absolute/path/to/dataset
```

Use fewer folds or a smaller batch on limited hardware:

```bash
FOLDS=3 BATCH_SIZE=4 DEVICE=cpu ./run_experiment.sh /absolute/path/to/dataset
```

## Multi-GPU execution

For this small frozen-feature benchmark, run one independent model per GPU
instead of using DDP for one model. With four GPUs:

```bash
GPUS=0,1,2,3 ./run_multi_gpu.sh \
  "/absolute/path/to/Dental OPG (Classification)"
```

With two GPUs, the script automatically runs the four models in two waves:

```bash
GPUS=0,1 BATCH_SIZE=16 ./run_multi_gpu.sh \
  "/absolute/path/to/Dental OPG (Classification)"
```

Each worker writes to a separate directory. Combined results are written to:

```text
outputs_binary_multi_gpu/
├── logs/
├── workers/
└── merged/
    ├── splits.csv
    ├── fold_metrics.csv
    ├── predictions.csv
    └── summary.csv
```

For the optional six-class benchmark:

```bash
CLASSES="BDC-BDR,Caries,Fractured Teeth,Healthy Teeth,Impacted Teeth,Infection" \
GPUS=0,1,2,3 \
./run_multi_gpu.sh "/absolute/path/to/Dental OPG (Classification)" \
  outputs_sixclass_multi_gpu
```

## Manifest mode

Example `manifest.csv`:

```csv
image_path,label,patient_id
healthy/a.jpg,healthy,P001
caries/b.jpg,caries,P002
```

Run directly:

```bash
.venv/bin/python run_feature_benchmark.py \
  --data-dir /absolute/path/to/dataset \
  --manifest /absolute/path/to/manifest.csv \
  --group-column patient_id \
  --output-dir outputs_grouped \
  --models resnet18 dinov3_vits16 siglip2 mambavision_t \
  --classes healthy caries
```


```bash
nohup env \
GPUS=0,1,2,3 \
MODELS="resnet18 dinov3_vits16 siglip2 mambavision_t" \
./run_multi_gpu.sh \
"data/Dental OPG XRAY Dataset/Dental OPG (Classification)" \
"outputs_binary_multi_gpu" > run_multi_gpu.log 2>&1 &

```