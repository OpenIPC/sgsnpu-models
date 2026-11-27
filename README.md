# sgsnpu-models

Object detectors compiled for the NPU ("IPU") of SigmaStar's infinity6c, the
SSC377x and SSC378x, for majestic's object detection on OpenIPC firmware.

| model | input | classes | IPU time on an SSC378DE |
|---|---|---|---|
| `yolov8n.img` | 640x384 NV12 | person, bicycle, car, motorcycle, bus, truck, cat, dog | 35-38 ms |
| `pvdn.img` | 640x352 NV12 | person, vehicle | 31 ms |

They are compiled from the checkpoints, and scored the way, of
[OpenIPC/xmnpu-models](https://github.com/OpenIPC/xmnpu-models), which trains
them and compiles them for another NPU. `export.py`, `operating_points.py`,
`images.txt` and `fetch.sh` are that repository's, at `902b00b`.

Each model is exported without its decode/NMS head: one NV12 picture in (BT.709
full range, converted to RGB inside the model), and six int16 NHWC outputs, a
64-channel DFL box map and a class map per stride (8, 16, 32), each with a scale
and no offset. Decoding is the consumer's job. `<model>.json` beside a model
names its class channels and the presets' thresholds, as xmnpu-models' README
describes.

## Licence plates

| file | input | what | IPU time on an SSC378DE |
|---|---|---|---|
| `lpr_spotter.img` | 416x256 NV12 | finds plates as oriented boxes (stride 8 and 16) | the three, 17-19 ms a plate |
| `lpr_stn.img` | 130x70 YUV | the 20 control points that straighten a plate | |
| `lpr_rec.img` | 128x64 YUV | the recognizer's features and their attention projection | |
| `lpr_dec.bin` | | the attention decoder, float32, run on the CPU | |

For majestic's licence-plate reading on infinity6c (its `lpr` settings). A
plate recognizer (thin-plate-spline transformer, convolutional encoder,
attention decoder) does not run on the IPU whole, so it comes as three parts:
the networks here, and between and after them the spline, the sampling and
the decoder, which majestic runs on the CPU (its `src/detect/lpr.c`;
`scripts/lpr.py` is the same arithmetic in numpy). The alphabet is that of
Russian and Soviet-era plates.

The sources are the exported networks, `lpr-onnx-v1.tar.gz` in
`sources.txt`; the networks were trained elsewhere and their training data
is not published. They are calibrated, and checked against FP32, on
synthetic plates (DejaVu Sans Condensed Bold) pasted into COCO128 pictures,
never on photographs of real plates (`scripts/lpr.sh`). On 2,000 real
plates the int8 pipeline reads 97.45% exactly, FP32 97.55%.
`lpr_eval.md` in a release is the check on the synthetic set.

## Releases

Only the `release` workflow publishes, never a workstation. A release carries
the models, their `.json`, `SHA256SUMS`, `eval.md` (int8 against FP32 on
held-out images; a model that loses a detection at 0.5 fails the release) and
`points.md` (what each preset achieves on COCO val2017). The int8 side of both
is the vendor's offline simulator, which gives the hardware's outputs bit for
bit: 96 of 96 tensors compared on an SSC378DE.

## Building

    pip install -r requirements-export.txt     # Python 3.11
    bash scripts/build.sh <toolchain.zip> out  # needs Docker
    bash scripts/points.sh out <val2017> <instances_val2017.json>

The compiler is SigmaStar's SGS_IPU_SDK vS1.1.6. It is confidential, is not in
this repository and cannot be had from it; `scripts/sdk.sh` pins its hash.
Everything else is pinned here. A rebuild gives the same bytes.

## License

AGPL-3.0, as the Ultralytics YOLOv8 weights the detectors are compiled
from; the licence-plate models are under the same licence.
