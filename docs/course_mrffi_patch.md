# Course MRFFI Patch

This patch adds a course-report aligned MobileMamba implementation without changing the upstream model file.

## Motivation

The submitted course report describes MobileMamba as a three-stage lightweight backbone with an MRFFI block that splits features into three branches:

1. **WTE-Mamba global branch** for long-range/global-frequency modeling.
2. **MK-DeConv local branch** with parallel `3x3` and `5x5` depthwise convolutions.
3. **Identity branch** for preserving untransformed channels.

The upstream implementation already contains the three-stage backbone and WTE-Mamba/wavelet branch, but its local branch is a single stage-wise depthwise kernel. This patch adds explicit MRFFI variants that implement the local branch as two parallel kernels while keeping the original implementation unchanged.

## Added files

- `model/mobilemamba_mrffi_course.py`
  - `MRFFIModule`: channel split into global/local/identity branches.
  - `MultiKernelDWConv2d_BN_ReLU`: local `3x3 + 5x5` depthwise branches.
  - `CourseMRFFIMobileMamba`: MobileMamba backbone using the explicit MRFFI block.
  - Registered variants:
    - `MobileMamba_T2_MRFFI`
    - `MobileMamba_T4_MRFFI`
    - `MobileMamba_S6_MRFFI`
    - `MobileMamba_B1_MRFFI`
    - `MobileMamba_B2_MRFFI`
    - `MobileMamba_B4_MRFFI`
- `configs/mobilemamba/mobilemamba_s6_mrffi_course.py`
- `configs/mobilemamba/mobilemamba_b4_mrffi_course.py`

## Usage

S6 training:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env \
  run.py -c configs/mobilemamba/mobilemamba_s6_mrffi_course -m train
```

B4 training:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --use_env \
  run.py -c configs/mobilemamba/mobilemamba_b4_mrffi_course -m train
```

S6 smoke-test import/forward:

```bash
python scripts/check_course_mrffi.py --model MobileMamba_S6_MRFFI --image-size 224 --batch-size 1
```

## Notes

- The original MobileMamba model names and files are not modified.
- Existing pretrained weights for the original models are not expected to load strictly into the new MRFFI variants because the local branch topology changes.
- The config keeps the default `300` epoch setting and exposes a `long_train` switch for the `1000` epoch training regime described in the report.
