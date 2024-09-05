!pip install easyocr
import easyocr
import cv2
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt

# Initialize EasyOCR reader (optimized for CPU, using quantized model)
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./models', download_enabled=True, quantize=True)

# Load the video
video_path = '2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Error opening video file: {video_path}")

frame_width = int(cap.get(3)) // 4  # Reduce frame size for faster processing
frame_height = int(cap.get(4)) // 4

# Performance metrics storage
fps_list_cpu, latency_list_cpu, cpu_usage_list, memory_usage_list = [], [], [], []
fps_list_gpu, latency_list_gpu = [], []
accuracy_list_cpu, accuracy_list_gpu = [], []
ocr_predictions_cpu, ocr_predictions_gpu = [], []  # Storing

# Preprocess frame doing ( grayscale and histogram equalization)
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return enhanced

# Function to run OCR on a single frame and calculate accuracy
def run_ocr_on_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    result = reader.readtext(preprocessed_frame)
    return result

# Calculate accuracy based on non-empty text detection
def calculate_accuracy(result):
    total_texts = len(result)
    if total_texts == 0:
        return 0  # If no text detected, return 0 accuracy
    correct_texts = sum(1 for r in result if r[1].strip())  # Count non-empty text detections
    accuracy = correct_texts / total_texts
    return accuracy

# Process video frames (simulating both CPU and GPU by running CPU OCR twice parallely )
def process_frame(frame):
    frame = cv2.resize(frame, (frame_width, frame_height))

    # CPU Mode
    start_time_cpu = time.time()
    result_cpu = run_ocr_on_frame(frame)
    end_time_cpu = time.time()

    fps_cpu = 1 / (end_time_cpu - start_time_cpu)
    latency_cpu = end_time_cpu - start_time_cpu
    accuracy_cpu = calculate_accuracy(result_cpu)

    # Simulated GPU Mode (also uses CPU for comparison purposes)
    start_time_gpu = time.time()
    result_gpu = run_ocr_on_frame(frame)
    end_time_gpu = time.time()

    fps_gpu = 1 / (end_time_gpu - start_time_gpu)
    latency_gpu = end_time_gpu - start_time_gpu
    accuracy_gpu = calculate_accuracy(result_gpu)

    # Return performance metrics and OCR results
    return result_cpu, fps_cpu, latency_cpu, result_gpu, fps_gpu, latency_gpu, accuracy_cpu, accuracy_gpu

frame_count = 0

# Main loop for reading video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video is done

    # Skip every second frame for faster processing
    if frame_count % 2 != 0:
        frame_count += 1
        continue

    # Process the frame
    result_cpu, fps_cpu, latency_cpu, result_gpu, fps_gpu, latency_gpu, accuracy_cpu, accuracy_gpu = process_frame(frame)

    # Store CPU performance metrics
    fps_list_cpu.append(fps_cpu)
    latency_list_cpu.append(latency_cpu)
    cpu_usage_list.append(psutil.cpu_percent(interval=0.05))
    memory_usage_list.append(psutil.virtual_memory().percent)
    accuracy_list_cpu.append(accuracy_cpu)

    # Store GPU (simulated) performance metrics
    fps_list_gpu.append(fps_gpu)
    latency_list_gpu.append(latency_gpu)
    accuracy_list_gpu.append(accuracy_gpu)

    # Store OCR predictions
    ocr_predictions_cpu.append(result_cpu)
    ocr_predictions_gpu.append(result_gpu)

    frame_count += 1

# Release video capture resources after processing
cap.release()

# Calculate average CPU and GPU performance metrics
avg_fps_cpu = np.mean(fps_list_cpu)
avg_latency_cpu = np.mean(latency_list_cpu)
avg_cpu_usage = np.mean(cpu_usage_list)
avg_memory_usage = np.mean(memory_usage_list)
avg_accuracy_cpu = np.mean(accuracy_list_cpu)

avg_fps_gpu = np.mean(fps_list_gpu)
avg_latency_gpu = np.mean(latency_list_gpu)
avg_accuracy_gpu = np.mean(accuracy_list_gpu)

# Combine all OCR results into a readable format
def compile_ocr_results(ocr_predictions):
    compiled_results = []
    for frame_index, frame_results in enumerate(ocr_predictions):
        compiled_results.append(f"\nFrame {frame_index + 1} OCR Results:")
        for detection in frame_results:
            bbox, text, confidence = detection
            compiled_results.append(f"Text: {text}, Confidence: {confidence}, Bounding Box: {bbox}")
    return "\n".join(compiled_results)

# Compile OCR results for both CPU and GPU (simulated) processing
compiled_results_cpu = compile_ocr_results(ocr_predictions_cpu)
compiled_results_gpu = compile_ocr_results(ocr_predictions_gpu)

# Save the OCR predictions to text files
with open('ocr_predictions_cpu.txt', 'w') as f:
    f.write(f"CPU OCR Results:\n{compiled_results_cpu}")

with open('ocr_predictions_gpu.txt', 'w') as f:
    f.write(f"GPU OCR Results (Simulated):\n{compiled_results_gpu}")

# Visualization: Performance Comparison Graph (CPU vs GPU)
metrics = ['FPS (CPU)', 'FPS (GPU)', 'CPU Usage (%)', 'Memory Usage (%)', 'Latency (CPU)', 'Latency (GPU)', 'Accuracy (CPU)', 'Accuracy (GPU)']
values = [avg_fps_cpu, avg_fps_gpu, avg_cpu_usage, avg_memory_usage, avg_latency_cpu, avg_latency_gpu, avg_accuracy_cpu * 100, avg_accuracy_gpu * 100]

x = np.arange(len(metrics))  # Label locations
width = 0.35  # Bar width

fig, ax = plt.subplots()
rects = ax.bar(x, values, width, label='Performance Metrics')

# Add labels, title, and x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_title('Raspberry Pi OCR Performance: CPU vs GPU (Simulated)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Function to label bars with values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3-point vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects)

fig.tight_layout()
plt.show()

# Final summary of performance metrics
print(f"Average FPS (CPU): {avg_fps_cpu}")
print(f"Average Latency (CPU): {avg_latency_cpu}")
print(f"Average CPU Usage: {avg_cpu_usage}%")
print(f"Average Memory Usage: {avg_memory_usage}%")
print(f"Average Accuracy (CPU): {avg_accuracy_cpu * 100:.2f}%")
print(f"Average FPS (GPU): {avg_fps_gpu}")
print(f"Average Latency (GPU): {avg_latency_gpu}")
print(f"Average Accuracy (GPU): {avg_accuracy_gpu * 100:.2f}%")

print("\nOCR Results saved to 'ocr_predictions_cpu.txt' and 'ocr_predictions_gpu.txt'.")
