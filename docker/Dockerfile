FROM python:3.7-slim-bookworm
RUN apt-get update -q && apt-get install -qy --no-install-recommends libglib2.0-0 libgomp1 >/dev/null
RUN pip install -q --no-cache-dir numpy==1.21.6 protobuf==3.20.3 six enum34==1.1.10 opencv-python-headless==4.5.5.64 \
  matplotlib==3.3.4 scipy==1.6.1 pillow joblib==1.0.1 pulp==2.4 sympy packaging pyyaml pycocotools tqdm psutil onnx==1.13.0 onnxruntime==1.14.1 onnxoptimizer==0.3.7
RUN pip install -q --no-cache-dir onnx-simplifier==0.4.17 || pip install -q --no-cache-dir --no-deps onnx-simplifier==0.4.17 rich
RUN pip install -q --no-cache-dir --no-deps tensorflow==1.14.0 && pip install -q --no-cache-dir absl-py astor gast==0.2.2 google-pasta grpcio keras-applications keras-preprocessing termcolor wrapt wheel tensorflow-estimator==1.14.0 "tensorboard<2" 
RUN pip install -q --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==1.8.0+cpu torchvision==0.9.0+cpu
RUN apt-get update -q && apt-get install -qy --no-install-recommends libtinfo5 libquadmath0 >/dev/null
