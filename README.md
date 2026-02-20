# Real-Time Object Detection on FPGA: Architecture and Optimizations

This repository contains the implementation, optimization, and deployment details for our hardware-accelerated object detection model. The project demonstrates a full-stack deployment, from model training and quantization to hardware acceleration and graphical user interface (GUI) integration.

## 📊 Model Performance & Resource Utilization

The following table summarizes the performance metrics and hardware resource utilization across our tested models:

| Model Version | Latency (ms) | Throughput (FPS) | Accuracy (mAP) | DSP Usage | BRAM | LUTs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model_v1_Baseline | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| Model_v2_Optimized| [Value] | [Value] | [Value] | [Value] | [Value] | [Value] |
| **Final_Deployed**| **[Value]** | **[Value]** | **[Value]** | **[Value]** | **[Value]** | **[Value]** |

## 📁 Model Files (`.pth` & `.onnx`)

All trained weights and exported computational graphs are located in the `models/` directory:
* **PyTorch Weights:** `models/pth/` (Contains pre-trained and fine-tuned `.pth` files).
* **ONNX Exports:** `models/onnx/` (Contains the exported `.onnx` files ready for compiler ingestion).

### LPyolo Wider Face ONNX
* **Status/Location:** [Specify if included or pending]
* **Details:** The `lpyolo_wider_face.onnx` model file is specifically trained for face detection using the Wider Face dataset. It is optimized for edge deployment.

---

## 🏗️ Hardware Acceleration & Deployment

### FINN vs. DPU
For FPGA deployment, we evaluated two primary Xilinx acceleration stacks:
* **DPU (Deep Learning Processor Unit):** An instruction-driven architecture optimized for standard CNN operations (typically INT8). It offers excellent general-purpose acceleration for standard architectures with minimal custom hardware design.
* **FINN:** A dataflow-style compiler that creates highly customized, streaming architectures tailored to specific neural networks. It excels at sub-byte (e.g., 1-bit, 4-bit) quantization and ultra-low latency, but requires more specialized model structures.
* **Conclusion:** [State which one you chose and a 1-sentence "why" - e.g., "We selected the DPU for its flexibility and ease of integrating our 8-bit quantized PyTorch models."]

### DPU Model Architecture
Our chosen architecture is tailored for the Xilinx DPU. The model graph was partitioned so that supported layers (Convolutions, ReLUs, Poolings) run entirely on the DPU, while unsupported custom operations fall back to the CPU.

#### Usage of DSP in DPU
The DPU heavily relies on the FPGA's Digital Signal Processor (DSP) slices (e.g., DSP48E2) to execute parallel Multiply-Accumulate (MAC) operations. Efficient use of DSPs directly translates to higher throughput. Our model architecture was explicitly designed to keep the DSP array fully saturated without bottlenecking the memory bandwidth.

### HLS Method
[If applicable:] We utilized High-Level Synthesis (HLS) in C++ to generate custom IP blocks for pre/post-processing steps that were a bottleneck for the CPU, seamlessly integrating them with the main acceleration pipeline.

### CPU Inference Time
For benchmarking, the baseline inference time running entirely on the embedded ARM CPU (without DPU acceleration) is:
* **CPU Latency:** `[XX] ms/frame` (~`[Y]` FPS)
* **Acceleration Factor:** The DPU provides a `[Z]x` speedup over the baseline CPU execution.

---

## 🛠️ Model Optimizations

To fit the model onto the edge device while maintaining real-time throughput, several optimization techniques were applied:

### 1. Architectural Changes
* **Depthwise vs. Pointwise Convolution:** We replaced standard convolutions with Depthwise Separable Convolutions. By decoupling spatial filtering (depthwise, $K \times K$) and channel combination (pointwise, $1 \times 1$), we drastically reduced both computational complexity (MACs) and parameter count.
* **Removed Fully Connected Layers:** The fully connected (dense) layers at the end of the network were removed, making the model a Fully Convolutional Network (FCN). This significantly reduced the memory footprint and allows the network to accept variable-sized input images.

### 2. Quantization
We implemented Post-Training Quantization (PTQ) / Quantization-Aware Training (QAT) to reduce the model's precision from FP32 to INT8. This is a strict requirement for DPU execution, reducing memory bandwidth requirements and allowing the use of integer-optimized DSP slices.

### 3. Pruning
We applied structural pruning to the network, eliminating filters with near-zero weights. This reduced the overall channel depth in less critical layers, directly decreasing the required computational resources.

### 4. Folding
* **Batch Normalization Folding:** During the ONNX export and compilation phase, Batch Normalization layers were mathematically folded into the preceding Convolutional layers. This eliminates the need to compute BN during inference, saving clock cycles and memory reads.

---

## 🖥️ System Integration

### GUI (Graphical User Interface)
The user interface is decoupled from the hardware. 
* **Backend:** A Flask server runs on the target board, serving the model predictions and handling hardware calls. 
* **Frontend:** The web UI is hosted remotely (e.g., via Vercel). 
* **Connection:** A secure tunnel (like Ngrok) bridges the local edge device to the public-facing frontend, allowing for remote monitoring and real-time visualization of the object detection feed.

### Video Driver for Webcam
We interface with the USB webcam using standard **V4L2 (Video4Linux2)** drivers on the embedded Linux environment. The OpenCV `VideoCapture(0)` module is optimized to grab frames efficiently and pass them to the DPU's memory space.

---

## 🎯 Datasets & Demos

### Model Selection for SAR / COCO
* **SAR (Synthetic Aperture Radar):** [Detail if you trained a specific model for SAR imagery].
* **COCO:** We utilized a subset of the COCO dataset for general object detection. Our model architecture was specifically selected to balance the high complexity of the COCO classes with the resource constraints of our edge device.

### COCO Dataset Video Demo
To run the live video demo using the COCO-trained model and print the detection outputs directly to the terminal, use the following script:

```bash
# Run the live video inference script
python3 run_coco_demo.py --model models/onnx/final_model.onnx --camera 0 --print_labels True
