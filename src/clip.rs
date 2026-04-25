use crate::guard;
use anyhow::Context;
use image::{imageops::FilterType, DynamicImage};
use rayon::iter::ParallelIterator;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ClipError {
    #[error("image not decodable: unsupported format and no embedded JPEG preview")]
    ImageNotDecodable,

    #[error("model file missing: {0}")]
    ModelNotFound(PathBuf),

    #[error("hub error for repo `{repo}`")]
    HubError {
        repo: String,
        #[source]
        source: hf_hub::api::sync::ApiError,
    },

    #[error("failed to load model: {reason}")]
    ModelLoadFailed {
        reason: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("tokenizer error")]
    TokenizerError(#[source] Box<dyn std::error::Error + Send + Sync>),
}

fn init_gpu() -> anyhow::Result<()> {
    let ok = ort::init()
        .with_execution_providers([ort::ep::ROCm::default().build()])
        .commit();
    anyhow::ensure!(
        ok,
        "ort::init() returned false — environment already initialised or EP registration failed"
    );
    Ok(())
}

/// Decode an image for CLIP inference.
///
/// The `image` crate handles JPEG/PNG/TIFF/etc. directly. For proprietary RAW
/// formats (RAF, NEF, CR2, ARW, …) it will fail, but those files almost always
/// contain an embedded full-resolution JPEG preview — we scan for the first
/// valid JPEG SOI marker and decode that instead.
#[tracing::instrument(skip_all)]
pub fn load_image_for_clip(bytes: &[u8]) -> Result<DynamicImage, ClipError> {
    if let Ok(img) = image::load_from_memory(bytes) {
        return Ok(img);
    }
    bytes
        .windows(2)
        .enumerate()
        .filter(|(_, w)| *w == [0xFF, 0xD8])
        .find_map(|(offset, _)| image::load_from_memory(&bytes[offset..]).ok())
        .ok_or(ClipError::ImageNotDecodable)
}

use hf_hub::api::sync::Api;
use ndarray::{Array4, Axis};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::IntoParallelRefIterator;
use tokenizers::Tokenizer;
use tracing::warn;

pub struct ClipTaggerManager;

impl deadpool::managed::Manager for ClipTaggerManager {
    type Type = ClipTagger;
    type Error = anyhow::Error;

    async fn create(&self) -> Result<ClipTagger, anyhow::Error> {
        Ok(tokio::task::spawn_blocking(ClipTagger::from_local_default).await??)
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

// TODO: here normally we could calculate the size of the vram if gpu is used
// and optimize
pub fn clip_pool(size: usize) -> Option<ClipTaggerPool> {
    if let Err(e) = init_gpu() {
        warn!("ROCm unavailable, falling back to CPU: {e:?}");
    }

    match deadpool::managed::Pool::builder(ClipTaggerManager)
        .max_size(size)
        .build()
    {
        Ok(pool) => {
            tracing::info!("tagger pool ready (max_size={size})");
            Some(pool)
        }
        Err(e) => {
            warn!("Tagger pool unavailable: {e:?}");
            None
        }
    }
}

pub struct ClipTagger {
    vision_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
}

impl ClipTagger {
    pub fn from_local_default() -> Result<Self, ClipError> {
        let vision_path = Path::new("models/clip-vit-base-patch32/onnx/vision_model.onnx");
        let text_path = Path::new("models/clip-vit-base-patch32/onnx/text_model.onnx");
        let tokenizer_path = Path::new("models/clip-vit-base-patch32/tokenizer.json");

        for path in [vision_path, text_path, tokenizer_path] {
            guard!(path.exists(), ClipError::ModelNotFound(path.to_path_buf()));
        }

        Self::new(vision_path, text_path, tokenizer_path)
    }

    pub fn from_hub_default() -> Result<Self, ClipError> {
        Self::from_hub("Xenova/clip-vit-base-patch32")
    }

    pub fn from_hub(repo_id: &str) -> Result<Self, ClipError> {
        let hub_err = |source| ClipError::HubError {
            repo: repo_id.to_string(),
            source,
        };

        let api = Api::new().map_err(&hub_err)?;
        let repo = api.model(repo_id.to_string());

        let vision_path = repo.get("onnx/vision_model.onnx").map_err(&hub_err)?;
        let text_path = repo.get("onnx/text_model.onnx").map_err(&hub_err)?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(&hub_err)?;

        Self::new(vision_path, text_path, tokenizer_path)
    }

    pub fn new(
        vision_path: impl AsRef<Path>,
        text_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, ClipError> {
        let vision_session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&vision_path))
            .map_err(|e| ClipError::ModelLoadFailed {
                reason: format!("vision model at {}", vision_path.as_ref().display()),
                source: Box::new(e),
            })?;
        let text_session = Session::builder()
            .and_then(|mut b| b.commit_from_file(&text_path))
            .map_err(|e| ClipError::ModelLoadFailed {
                reason: format!("text model at {}", text_path.as_ref().display()),
                source: Box::new(e),
            })?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref()).map_err(|e| {
            ClipError::ModelLoadFailed {
                reason: format!("tokenizer at {}", tokenizer_path.as_ref().display()),
                source: e,
            }
        })?;

        Ok(Self {
            vision_session,
            text_session,
            tokenizer,
        })
    }

    pub fn from_memory(
        vision_model: &[u8],
        text_model: &[u8],
        tokenizer_json: &[u8],
    ) -> Result<Self, ClipError> {
        let vision_session = Session::builder()
            .and_then(|mut b| b.commit_from_memory(vision_model))
            .map_err(|e| ClipError::ModelLoadFailed {
                reason: "vision model from memory".into(),
                source: Box::new(e),
            })?;
        let text_session = Session::builder()
            .and_then(|mut b| b.commit_from_memory(text_model))
            .map_err(|e| ClipError::ModelLoadFailed {
                reason: "text model from memory".into(),
                source: Box::new(e),
            })?;
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_json).map_err(|e| ClipError::ModelLoadFailed {
                reason: "tokenizer from memory".into(),
                source: e,
            })?;

        Ok(Self {
            vision_session,
            text_session,
            tokenizer,
        })
    }

    #[cfg(feature = "embed-models")]
    pub fn from_embedded() -> Result<Self, ClipError> {
        use rust_embed::RustEmbed;

        #[derive(RustEmbed)]
        #[folder = "models/clip-vit-base-patch32/"]
        struct Asset;

        let vision_model = Asset::get("onnx/vision_model.onnx").expect(
            "BUG: vision_model.onnx missing from embedded assets — rebuild with models present",
        );
        let text_model = Asset::get("onnx/text_model.onnx").expect(
            "BUG: text_model.onnx missing from embedded assets — rebuild with models present",
        );
        let tokenizer = Asset::get("tokenizer.json").expect(
            "BUG: tokenizer.json missing from embedded assets — rebuild with models present",
        );

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

        let batch_size = images.len();

        // 1. Preprocess all images and stack into [batch, 3, 224, 224]
        let preprocessed: Vec<ndarray::Array3<f32>> = images
            .par_iter()
            .map(|img| self.preprocess_image(img))
            .collect();

        let preprocessed = Array4::from_shape_vec((batch_size, 3, 244, 244), preprocessed).unwrap();

        let pixel_tensor =
            Tensor::from_array(preprocessed).expect("BUG: pixel_values shape is statically known");

        // 2. Vision model → image embeddings [batch, embed_dim]
        let image_outputs = self
            .vision_session
            .run(ort::inputs!["pixel_values" => pixel_tensor.view()])
            .context("CLIP vision inference failed")?;
        let image_embeds_raw = image_outputs
            .get("image_embeds")
            .expect("BUG: CLIP vision model must produce 'image_embeds' output");
        let (image_shape, image_data) = image_embeds_raw
            .try_extract_tensor::<f32>()
            .expect("BUG: image_embeds must be f32 tensor");
        let image_embeds = ndarray::ArrayView2::from_shape(
            (image_shape[0] as usize, image_shape[1] as usize),
            image_data,
        )
        .expect("BUG: image_embeds shape must match extracted dimensions");

        // 3. Text model → text embeddings [num_tags, embed_dim]
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        let encoded = tokenizer
            .encode_batch(tags.to_vec(), true)
            .map_err(|e| anyhow::anyhow!(e))
            .context("tokenizer failed to encode tags")?;

        let num_tags = tags.len();
        let seq_len = encoded[0].get_ids().len();
        let mut input_ids = Vec::with_capacity(num_tags * seq_len);
        for enc in &encoded {
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
        }
        let input_ids_tensor = Tensor::from_array(([num_tags, seq_len], input_ids))
            .expect("BUG: input_ids shape is deterministic");

        let text_outputs = self
            .text_session
            .run(ort::inputs!["input_ids" => input_ids_tensor.view()])
            .context("CLIP text inference failed")?;
        let text_embeds_raw = text_outputs
            .get("text_embeds")
            .expect("BUG: CLIP text model must produce 'text_embeds' output");
        let (text_shape, text_data) = text_embeds_raw
            .try_extract_tensor::<f32>()
            .expect("BUG: text_embeds must be f32 tensor");
        let text_embeds = ndarray::ArrayView2::from_shape(
            (text_shape[0] as usize, text_shape[1] as usize),
            text_data,
        )
        .expect("BUG: text_embeds shape must match extracted dimensions");

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
    pub fn predict(
        &mut self,
        image: &DynamicImage,
        tags: &[String],
        threshold: f32,
    ) -> anyhow::Result<Vec<String>> {
        Ok(self
            .predict_batch(std::slice::from_ref(image), tags, threshold)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// Prepare an image for CLIP inference.
    ///
    /// 1. Resize to fit within 224x224 (preserving aspect ratio).
    /// 2. Center-pad onto a black 224x224 canvas.
    /// 3. Convert to `f32`, scale to `[0, 1]`, then normalize per-channel with
    ///    the OpenAI CLIP mean/std `([0.481, 0.458, 0.408], [0.269, 0.261, 0.276])`.
    ///
    /// Returns a `[3, 224, 224]` tensor in CHW layout.
    fn preprocess_image(&self, image: &DynamicImage) -> ndarray::Array3<f32> {
        let (target_width, target_height) = (224u32, 224u32);

        let resized = image.resize(target_width, target_height, FilterType::Triangle);
        let (w, h) = (resized.width(), resized.height());

        let mut canvas = image::ImageBuffer::new(target_width, target_height);
        let x_offset = (target_width - w) / 2;
        let y_offset = (target_height - h) / 2;
        image::imageops::overlay(
            &mut canvas,
            &resized.to_rgb8(),
            x_offset.into(),
            y_offset.into(),
        );

        // [H, W, 3] → f32 → [3, H, W]
        let array = ndarray::Array3::from_shape_vec(
            (target_height as usize, target_width as usize, 3),
            canvas.as_raw().clone(),
        )
        .expect("BUG: canvas is 224x224 RGB8 — shape is deterministic");
        let array = array.mapv(|x| x as f32 / 255.0);
        let array = array.permuted_axes([2, 0, 1]); // [3, 224, 224]

        let mean = ndarray::array![[[0.48145466f32, 0.4578275, 0.40821073]]];
        let std = ndarray::array![[[0.26862954f32, 0.26130258, 0.27577711]]];

        (array - mean) / std
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
        let tags = vec![
            "a photo of a cat".to_string(),
            "a photo of a dog".to_string(),
            "a white image".to_string(),
        ];
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
        let tags = vec![
            "a wide red stripe".to_string(),
            "a tall blue stripe".to_string(),
            "a red square".to_string(),
        ];
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
