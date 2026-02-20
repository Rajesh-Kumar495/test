# Real-Time Object Detection on FPGA: PYNQ-Z2 Acceleration

This repository details the implementation, optimization, and deployment of hardware-accelerated object detection models on the Xilinx PYNQ-Z2 board. It documents our architectural journey through different Xilinx toolchains to balance resource constraints, model complexity, and inference speed.

## 📁 Model Directory (`.pth` & `.onnx`)

All trained weights and exported computational graphs are located in the `models/` directory. This allows for easy tracking of model versions from PyTorch training to hardware compilation:
* **PyTorch Weights (`models/pth/`):** Contains the pre-trained and fine-tuned `.pth` files.
* **ONNX Exports (`models/onnx/`):** Contains the exported `.onnx` files ready for the compiler.
    * *Includes:* `lpyolo_wider_face.onnx` – A lightweight YOLO model specifically trained on the WiderFace dataset, highly optimized for edge deployment.

---

## 🏗️ Hardware Acceleration Journey

Our deployment strategy evolved through three distinct phases to overcome the specific resource constraints of the PYNQ-Z2 board.

### 1. The HLS Method (Initial Approach)
We initially attempted to accelerate custom operations using High-Level Synthesis (C++ to RTL). However, the generated IP blocks resulted in extremely high and unoptimized resource utilization on the FPGA, making it impossible to fit a complete, efficient pipeline. This bottleneck pushed us toward dedicated neural network compilers.

### 2. Xilinx FINN Toolchain

To solve the resource optimization issues of HLS, we transitioned to the FINN dataflow compiler. FINN creates highly customized, streaming architectures tailored specifically to the neural network being compiled. 
* **Implementations:** We successfully implemented SAR (Synthetic Aperture Radar) detection, face detection using a low-performance YOLO variant, and basic 3-4 layer convolutional networks.
* **Folding Optimization:** To satisfy the strict resource constraints of the PYNQ-Z2 while maintaining performance, we heavily utilized FINN's **Folding** technique. Folding allows the time-multiplexing of compute resources (reusing hardware blocks for multiple operations), balancing the trade-off between throughput and FPGA slice utilization.
* **The Bottleneck:** FINN is highly pipelined, but we hit a hard wall regarding model depth. The PYNQ-Z2 lacks the logic fabric to support models with a high number of layers in a spatial FINN architecture. Our maximum achievable accuracy on the COCO dataset was only 10 mAP. Furthermore, the FINN toolchain lacked support for crucial modern layers like Max Pooling and Depthwise Convolutions.

### 3. Xilinx DPU Framework

Because we needed to run huge models like YOLOv3 to achieve acceptable accuracy, we moved to the Deep Learning Processor Unit (DPU) framework. 
* **DPU Architecture & DSP Usage:** The DPU is a soft-core processor overlaid on the FPGA, executing neural networks via an instruction set rather than generating custom RTL per model. It heavily utilizes the FPGA's Digital Signal Processor (DSP) slices to execute parallel Multiply-Accumulate (MAC) operations, acting as the primary computation engine for convolutions.
* **Advantages:** Unlike FINN, the DPU seamlessly supports deep architectures and operations like Depthwise Convolutions.
* **The Trade-off:** While the DPU allowed us to deploy YOLOv3, the latency and throughput were significantly worse on the PYNQ-Z2 compared to the theoretical speed of FINN's spatial architecture, due to the DPU's memory-bound, instruction-driven nature.

### ⚖️ FINN vs. DPU Summary

| Feature | FINN | DPU |
| :--- | :--- | :--- |
| **Architecture** | Spatial Dataflow (Custom RTL per model) | Instruction-driven Overlay |
| **Model Size** | Limited to shallow models (3-4 conv layers) | Supports huge, deep models (YOLOv3) |
| **Layer Support** | Lacked Max Pool / Depthwise Conv | Broad support (Depthwise, Max Pool) |
| **Throughput** | High (if the model fits) | Very poor (High latency on PYNQ-Z2) |

---

## 🛠️ Model Optimizations

To maximize the efficiency of the models running on the target hardware, we applied several aggressive optimization techniques:

### Architectural Changes
* **Depthwise vs. Standard Convolution:** To combat latency in the DPU, we replaced standard convolutions with Depthwise Convolutions. Standard convolutions mix spatial and channel information simultaneously, requiring massive parameter counts. Depthwise convolutions process each channel separately before combining them, drastically reducing both the parameter count and the inference latency, which the DPU handles efficiently.
* **Removed Fully Connected Layers:** All dense layers at the network's tail were removed to create a Fully Convolutional Network, significantly reducing the memory footprint.

### Quantization & Pruning
* **Quantization:** We utilized Quantization-Aware Training (QAT). For our face detection model trained on the WiderFace dataset (which contains only one class and requires fewer weights compared to COCO), we aggressively quantized the model down to **8-bit weights and 3-bit activations**.
* **Pruning:** During the QAT process, we applied L1 Regularization. This technique forces less important weights toward exactly zero, effectively pruning the network during training and skipping redundant computations during hardware inference.

---

## 🖥️ System Integration & GUI

To make the hardware-accelerated model accessible to users without technical knowledge, we decoupled the user interface from the edge device.

* **Video Input:** The webcam feed is captured locally on the user's Laptop/PC via OpenCV, utilizing the PC's GPU for efficient frame grabbing and preprocessing.
* **Network Protocol:** The processed frames are streamed from the Laptop/PC directly to the PYNQ-Z2 board over a standard **TCP Protocol**.
* **Inference & Response:** The PYNQ-Z2 receives the frame, executes the accelerated inference via the DPU/FINN, and sends only the resulting bounding box coordinates back over TCP. 
* **GUI Overlay:** The PC receives the coordinates and draws the bounding boxes on the live video feed.

---

## 📊 Hardware Performance & Resource Utilization

| Model Version | Target | Latency (ms) | Throughput (FPS) | Accuracy | DSP Usage | BRAM | LUTs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Basic CNN (3-4 Layers) | FINN | [Value] | [Value] | 10 mAP (COCO)| [Value] | [Value] | [Value] |
| SAR Detection | FINN | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| LPyolo (WiderFace) | FINN/DPU| [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| YOLOv3 | DPU | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |

*Note: Baseline CPU inference time running entirely on the embedded ARM Cortex-A9 (without hardware acceleration) is **[XX] ms/frame**, highlighting the necessity of the PL fabric for real-time applications.*

---

## 🎯 COCO Dataset Video Demo

To run the live video demo using the COCO-trained model and print the detection outputs directly to the terminal, use the following command:

```bash
# Run the live video inference script
python3 run_coco_demo.py --model models/onnx/yolov3_coco.onnx --camera 0 --print_labels True
