# Download CLIP models from Hugging Face
download-models:
	@mkdir -p models/clip-vit-base-patch32/onnx || true # We don't care the folder is already there
	@echo "Downloading CLIP vision model..."
	curl -L https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx -o models/clip-vit-base-patch32/onnx/vision_model.onnx
	@echo "Downloading CLIP text model..."
	curl -L https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model.onnx -o models/clip-vit-base-patch32/onnx/text_model.onnx
	@echo "Downloading CLIP tokenizer..."
	curl -L https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json -o models/clip-vit-base-patch32/tokenizer.json
	@echo "Download complete."

# Coverage test against exiftool.org sample archives + a few RAW samples.
# Downloads ~50 MB on first run, caches under target/tmp/.
test-samples:
	cargo test --test sample_images -- --ignored --nocapture
