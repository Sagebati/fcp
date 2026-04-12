use anyhow::Context;
use std::path::Path;
use image::{DynamicImage, imageops::FilterType};

pub fn init_gpu() -> anyhow::Result<()> {
    let ok = ort::init()
        .with_execution_providers([ort::ep::ROCm::default().build()])
        .commit();
    anyhow::ensure!(ok, "ort::init() returned false — environment already initialised or EP registration failed");
    Ok(())
}

/// Decode an image for CLIP inference.
///
/// The `image` crate handles JPEG/PNG/TIFF/etc. directly. For proprietary RAW
/// formats (RAF, NEF, CR2, ARW, …) it will fail, but those files almost always
/// contain an embedded full-resolution JPEG preview — we scan for the first
/// valid JPEG SOI marker and decode that instead.
#[tracing::instrument(skip_all)]
pub fn load_image_for_clip(bytes: &[u8]) -> anyhow::Result<DynamicImage> {
    if let Ok(img) = image::load_from_memory(bytes) {
        return Ok(img);
    }
    bytes
        .windows(2)
        .enumerate()
        .filter(|(_, w)| *w == [0xFF, 0xD8])
        .find_map(|(offset, _)| image::load_from_memory(&bytes[offset..]).ok())
        .context("no decodable image: unsupported format and no embedded JPEG preview found")
}

use ort::session::Session;
use ort::value::Tensor;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use ndarray::Axis;

pub struct ClipTaggerManager;

impl deadpool::managed::Manager for ClipTaggerManager {
    type Type = ClipTagger;
    type Error = anyhow::Error;

    async fn create(&self) -> Result<ClipTagger, anyhow::Error> {
        tokio::task::spawn_blocking(ClipTagger::from_local_default).await?
    }

    async fn recycle(
        &self,
        _obj: &mut ClipTagger,
        _metrics: &deadpool::managed::Metrics,
    ) -> deadpool::managed::RecycleResult<anyhow::Error> {
        Ok(())
    }
}

pub type ClipTaggerPool = deadpool::managed::Pool<ClipTaggerManager>;

pub struct ClipTagger {
    vision_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
}

impl ClipTagger {
    pub fn from_local_default() -> anyhow::Result<Self> {
        let vision_path = Path::new("models/clip-vit-base-patch32/onnx/vision_model.onnx");
        let text_path = Path::new("models/clip-vit-base-patch32/onnx/text_model.onnx");
        let tokenizer_path = Path::new("models/clip-vit-base-patch32/tokenizer.json");

        Self::new(vision_path, text_path, tokenizer_path)
    }

    pub fn from_hub_default() -> anyhow::Result<Self> {
        Self::from_hub("Xenova/clip-vit-base-patch32")
    }

    pub fn from_hub(repo_id: &str) -> anyhow::Result<Self> {
        let api = Api::new()?;
        let repo = api.model(repo_id.to_string());

        let vision_path = repo.get("onnx/vision_model.onnx")?;
        let text_path = repo.get("onnx/text_model.onnx")?;
        let tokenizer_path = repo.get("tokenizer.json")?;

        Self::new(vision_path, text_path, tokenizer_path)
    }

    pub fn new(vision_path: impl AsRef<Path>, text_path: impl AsRef<Path>, tokenizer_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let vision_session = Session::builder()?.commit_from_file(vision_path)?;
        let text_session = Session::builder()?.commit_from_file(text_path)?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { vision_session, text_session, tokenizer })
    }

    pub fn from_memory(vision_model: &[u8], text_model: &[u8], tokenizer_json: &[u8]) -> anyhow::Result<Self> {
        let vision_session = Session::builder()?.commit_from_memory(vision_model)?;
        let text_session = Session::builder()?.commit_from_memory(text_model)?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { vision_session, text_session, tokenizer })
    }

    #[cfg(feature = "embed-models")]
    pub fn from_embedded() -> anyhow::Result<Self> {
        use rust_embed::RustEmbed;

        #[derive(RustEmbed)]
        #[folder = "models/clip-vit-base-patch32/"]
        struct Asset;

        let vision_model = Asset::get("onnx/vision_model.onnx")
            .ok_or_else(|| anyhow::anyhow!("Vision model not found in embedded assets"))?;
        let text_model = Asset::get("onnx/text_model.onnx")
            .ok_or_else(|| anyhow::anyhow!("Text model not found in embedded assets"))?;
        let tokenizer = Asset::get("tokenizer.json")
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not found in embedded assets"))?;

        Self::from_memory(&vision_model.data, &text_model.data, &tokenizer.data)
    }

    /// Run inference on a batch of images against a set of candidate tags.
    ///
    /// Returns one `Vec<String>` per image — the tags whose cosine similarity
    /// to the image embedding exceeds `threshold`.
    pub fn predict_batch(
        &mut self,
        images: &[DynamicImage],
        tags: &[String],
        threshold: f32,
    ) -> anyhow::Result<Vec<Vec<String>>> {
        if images.is_empty() || tags.is_empty() {
            return Ok(vec![vec![]; images.len()]);
        }

        // 1. Preprocess all images and stack into [batch, 3, 224, 224]
        let preprocessed: Vec<ndarray::Array3<f32>> = images
            .iter()
            .map(|img| self.preprocess_image(img))
            .collect::<anyhow::Result<_>>()?;
        let views: Vec<_> = preprocessed.iter().map(|a| a.view()).collect();
        let pixel_values = ndarray::stack(Axis(0), &views)?;
        let pixel_tensor = Tensor::from_array(pixel_values)?;

        // 2. Vision model → image embeddings [batch, embed_dim]
        let image_outputs = self.vision_session.run(ort::inputs!["pixel_values" => pixel_tensor.view()])?;
        let image_embeds_raw = image_outputs.get("image_embeds")
            .ok_or_else(|| anyhow::anyhow!("Failed to get image_embeds"))?;
        let (image_shape, image_data) = image_embeds_raw.try_extract_tensor::<f32>()?;
        let image_embeds = ndarray::ArrayView2::from_shape(
            (image_shape[0] as usize, image_shape[1] as usize),
            image_data,
        )?;

        // 3. Text model → text embeddings [num_tags, embed_dim]
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        let encoded = tokenizer.encode_batch(tags.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        let num_tags = tags.len();
        let seq_len = encoded[0].get_ids().len();
        let mut input_ids = Vec::with_capacity(num_tags * seq_len);
        for enc in &encoded {
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
        }
        let input_ids_tensor = Tensor::from_array(([num_tags, seq_len], input_ids))?;

        let text_outputs = self.text_session.run(ort::inputs!["input_ids" => input_ids_tensor.view()])?;
        let text_embeds_raw = text_outputs.get("text_embeds")
            .ok_or_else(|| anyhow::anyhow!("Failed to get text_embeds"))?;
        let (text_shape, text_data) = text_embeds_raw.try_extract_tensor::<f32>()?;
        let text_embeds = ndarray::ArrayView2::from_shape(
            (text_shape[0] as usize, text_shape[1] as usize),
            text_data,
        )?;

        // 4. Normalise both embedding matrices
        let img_norms = image_embeds.map_axis(Axis(1), |row| row.dot(&row).sqrt() + 1e-6);
        let image_norm = &image_embeds / &img_norms.insert_axis(Axis(1));

        let txt_norms = text_embeds.map_axis(Axis(1), |row| row.dot(&row).sqrt() + 1e-6);
        let text_norm = &text_embeds / &txt_norms.insert_axis(Axis(1));

        // 5. Similarity matrix [batch, num_tags] = image_norm @ text_norm.T
        let similarities = image_norm.dot(&text_norm.t());

        // 6. For each image collect tags above threshold
        Ok(similarities
            .outer_iter()
            .map(|row| {
                row.iter()
                    .zip(tags.iter())
                    .filter(|(&s, _)| s >= threshold)
                    .map(|(_, tag)| tag.clone())
                    .collect()
            })
            .collect())
    }

    /// Convenience wrapper for a single image.
    pub fn predict(&mut self, image: &DynamicImage, tags: &[String], threshold: f32) -> anyhow::Result<Vec<String>> {
        Ok(self.predict_batch(std::slice::from_ref(image), tags, threshold)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    fn preprocess_image(&self, image: &DynamicImage) -> anyhow::Result<ndarray::Array3<f32>> {
        let (target_width, target_height) = (224u32, 224u32);

        let resized = image.resize(target_width, target_height, FilterType::Triangle);
        let (w, h) = (resized.width(), resized.height());

        let mut canvas = image::ImageBuffer::new(target_width, target_height);
        let x_offset = (target_width - w) / 2;
        let y_offset = (target_height - h) / 2;
        image::imageops::overlay(&mut canvas, &resized.to_rgb8(), x_offset.into(), y_offset.into());

        // [H, W, 3] → f32 → [3, H, W]
        let array = ndarray::Array3::from_shape_vec(
            (target_height as usize, target_width as usize, 3),
            canvas.as_raw().clone(),
        )?;
        let array = array.mapv(|x| x as f32 / 255.0);
        let mut array = array.permuted_axes([2, 0, 1]); // [3, 224, 224]

        let mean = ndarray::array![0.48145466f32, 0.4578275, 0.40821073];
        let std  = ndarray::array![0.26862954f32, 0.26130258, 0.27577711];
        for i in 0..3 {
            let mut ch = array.index_axis_mut(Axis(0), i);
            ch -= mean[i];
            ch /= std[i];
        }

        Ok(array) // [3, 224, 224]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    #[cfg(feature = "embed-models")]
    fn test_clip_tagger_embedded() -> anyhow::Result<()> {
        let mut tagger = ClipTagger::from_embedded()?;
        let img = DynamicImage::new_rgb8(224, 224);
        let tags = vec!["a white image".to_string()];
        let result = tagger.predict(&img, &tags, 0.2)?;
        assert!(result.contains(&"a white image".to_string()));
        Ok(())
    }

    #[test]
    fn test_clip_tagger_local() -> anyhow::Result<()> {
        let mut tagger = ClipTagger::from_local_default()?;
        let img = DynamicImage::new_rgb8(224, 224);
        let tags = vec!["a photo of a cat".to_string(), "a photo of a dog".to_string(), "a white image".to_string()];
        let result = tagger.predict(&img, &tags, 0.2)?;
        println!("Tags: {:?}", result);
        Ok(())
    }

    #[test]
    fn test_clip_tagger_aspect_ratio() -> anyhow::Result<()> {
        let mut tagger = ClipTagger::from_local_default()?;
        let mut img = image::ImageBuffer::new(1000, 200);
        for x in 0..1000 {
            for y in 0..200 {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        let img = DynamicImage::ImageRgb8(img);
        let tags = vec!["a wide red stripe".to_string(), "a tall blue stripe".to_string(), "a red square".to_string()];
        let result = tagger.predict(&img, &tags, 0.1)?;
        println!("Tags for wide image: {:?}", result);
        assert!(result.contains(&"a wide red stripe".to_string()));
        Ok(())
    }

    #[test]
    fn test_clip_tagger_from_memory() -> anyhow::Result<()> {
        let vision_bytes = std::fs::read("models/clip-vit-base-patch32/onnx/vision_model.onnx")?;
        let text_bytes = std::fs::read("models/clip-vit-base-patch32/onnx/text_model.onnx")?;
        let tokenizer_bytes = std::fs::read("models/clip-vit-base-patch32/tokenizer.json")?;
        let mut tagger = ClipTagger::from_memory(&vision_bytes, &text_bytes, &tokenizer_bytes)?;
        let img = DynamicImage::new_rgb8(224, 224);
        let tags = vec!["a white image".to_string()];
        let result = tagger.predict(&img, &tags, 0.2)?;
        assert!(result.contains(&"a white image".to_string()));
        Ok(())
    }
}
