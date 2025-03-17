## Steps
1. `preprocess_images_to_datasets.py`
	- UPDATE BASED ON YOUR AI MODEL and file paths
2. `inference_image_datasets.py`
	- UPDATE BASED ON YOUR AI MODEL and file paths
3. `process_results_datasets.py`
	- UPDATE BASED ON YOUR AI MODEL and file paths
4. `calculate_accuracy.py`
	- UPDATE BASED ON YOUR AI MODEL and file paths

Checkout the helper scripts for splitting the ImageNet images into
50 folders with 1000 images each. I did this so we could test different
size datasets. 10k, 20k, 2k, etc.

- [ ] ResNeXt50
- [ ] ResNeXt50Quantized
- [X] Shufflenet-v2
- [ ] Shufflenet-v2-Quantized
- [X] SqueezeNet-1_1
- [ ] SqueezeNet-1_1Quantized
- [X] Swin-Base
- [X] Swin-Small
- [X] Swin-Tiny
- [X] VIT
- [X] WideResNet50
- [ ] WideResNet50-Quantized