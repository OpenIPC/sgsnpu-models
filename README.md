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

AGPL-3.0, as the Ultralytics YOLOv8 weights the models are compiled from.
