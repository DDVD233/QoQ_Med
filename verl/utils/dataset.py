# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import logging
import traceback
import wfdb
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
from scipy.signal import resample

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import copy

# For video processing
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('dataset_worker.log'), logging.StreamHandler()]
)
logger = logging.getLogger('RLHFDataset')


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    # Handle segmentation masks separately
    seg_masks = []
    max_height, max_width = 0, 0

    # First pass: collect all tensors and find max dimensions for segmentation masks
    for feature in features:
        for key, value in feature.items():
            if key == "segmentation_mask":
                assert isinstance(value, np.ndarray)
                if len(value.shape) == 3:
                    c, h, w = value.shape
                    # Update max dimensions
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                elif len(value.shape) == 2:
                    h, w = value.shape
                    max_height = max(max_height, h)
                    max_width = max(max_width, w)
                seg_masks.append(value)
            elif isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                if key == "time-series":
                    continue
                non_tensors[key].append(value)

    # Second pass: pad segmentation masks to max dimensions
    padded_masks = []
    for mask in seg_masks:
        if mask is None:
            # Create zero array for missing segmentation masks
            padded_masks.append(np.zeros((1, max_height, max_width), dtype=np.float32))
        else:
            # Get current dimensions
            if len(mask.shape) == 3:  # [C, H, W]
                c, h, w = mask.shape
                # Calculate padding (bottom, right)
                pad_bottom = max_height - h
                pad_right = max_width - w
                # Pad the mask using numpy padding
                padded_mask = np.pad(mask, ((0, 0), (0, pad_bottom), (0, pad_right)),
                                     mode='constant', constant_values=0)
                padded_masks.append(padded_mask)
            elif len(mask.shape) == 2:  # [H, W]
                h, w = mask.shape
                # Calculate padding (bottom, right)
                pad_bottom = max_height - h
                pad_right = max_width - w
                # Pad the mask using numpy padding
                padded_mask = np.pad(mask, ((0, pad_bottom), (0, pad_right)),
                                     mode='constant', constant_values=0)
                padded_mask = padded_mask[np.newaxis, :, :]  # Add channel dimension
                padded_masks.append(padded_mask)
            else:
                # Handle unexpected shapes
                padded_masks.append(np.zeros((1, max_height, max_width), dtype=np.float32))

    # Add padded segmentation masks to non_tensors
    non_tensors["segmentation_mask"] = np.stack(padded_masks, axis=0)

    # Stack other tensors
    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    # Convert other non-tensors to arrays
    for key, value in non_tensors.items():
        if key != "segmentation_mask":  # We've already handled segmentation masks
            non_tensors[key] = np.array(value, dtype=object)

    # Combine tensors and non-tensors
    return {**tensors, **non_tensors}


def extract_video_frames(video_path: str, num_frames: int = 4) -> List[ImageObject]:
    """
    Extract a specified number of frames uniformly from a video.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract

    Returns:
        List of PIL Image objects
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            logger.error(f"Invalid frame count for video: {video_path}")
            return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]

        # Calculate frame indices to extract
        frames_to_extract = []
        if num_frames == 1:
            # Just get the middle frame
            frames_to_extract = [total_frames // 2]
        else:
            # Get frames uniformly distributed across the video
            for i in range(num_frames):
                frame_idx = int(i * (total_frames - 1) / (num_frames - 1)) if num_frames > 1 else 0
                frames_to_extract.append(frame_idx)

        # Extract the frames
        frames = []
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx} from video: {video_path}")
                frames.append(Image.new("RGB", (224, 224), (128, 128, 128)))
                continue

            # Convert from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

        cap.release()
        return frames

    except Exception as e:
        logger.error(f"Error extracting frames from video {video_path}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return placeholder images
        return [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(num_frames)]
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            logger.debug(f"Released video capture for {video_path}")


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        try:
            if self.max_pixels is not None and (image.width * image.height) > self.max_pixels:
                resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                logger.debug(
                    f"Resizing image from {image.width}x{image.height} to {width}x{height} (max_pixels: {self.max_pixels})")
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)

            if self.min_pixels is not None and (image.width * image.height) < self.min_pixels:
                resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                logger.debug(
                    f"Resizing image from {image.width}x{image.height} to {width}x{height} (min_pixels: {self.min_pixels})")
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)

            if image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            # Return a small fallback image instead of crashing
            fallback = Image.new("RGB", (224, 224), (128, 128, 128))
            return fallback

    def process_time_series(self, full_path: str) -> torch.Tensor:
        full_path = os.path.splitext(full_path)[0]
        record = wfdb.rdrecord(full_path)
        segment_length = 2500
        ecg_data = record.p_signal

        # Resample the data if the sampling frequency is not 500 Hz
        if record.fs != 500:
            # Calculate the new length for resampling
            new_length = int((500 / record.fs) * record.sig_len)
            ecg_data = resample(ecg_data, new_length)

        # Truncate if data is longer than segment_length
        if ecg_data.shape[0] > segment_length:
            ecg_data = ecg_data[:segment_length]

        # Pad with zeros if data is shorter than segment_length
        if ecg_data.shape[0] < segment_length:
            padding = np.zeros((segment_length - ecg_data.shape[0], ecg_data.shape[1]))
            ecg_data = np.vstack((ecg_data, padding))

        ecg_data = ecg_data.T
        ecg_data = ecg_data[[0, 1, 6, 7, 8, 9, 10, 11], :]
        return torch.tensor(ecg_data)


def resize_bbox(bbox, original_width, original_height, new_width, new_height):
    """
    Resize bounding box coordinates based on image resizing ratio.

    Args:
        bbox (list): Original bounding box in format [x_min, y_min, x_max, y_max]
        original_width (int): Width of the original image
        original_height (int): Height of the original image
        new_width (int): Width of the resized image
        new_height (int): Height of the resized image

    Returns:
        list: Resized bounding box coordinates
    """
    # Calculate scaling factors
    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    # Apply scaling to bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates
    new_x_min = x_min * width_ratio
    new_y_min = y_min * height_ratio
    new_x_max = x_max * width_ratio
    new_y_max = y_max * height_ratio

    return [new_x_min, new_y_min, new_x_max, new_y_max]


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            processor: Optional[ProcessorMixin],
            prompt_key: str = "prompt",
            answer_key: str = "answer",
            image_key: str = "images",
            time_series_key: str = "time-series",
            max_prompt_length: int = 1024,
            truncation: str = "error",
            format_prompt: Optional[str] = None,
            max_pixels: Optional[int] = None,
            min_pixels: Optional[int] = None,
            video_frames=2,
            filter_overlong_prompts: bool = True,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.time_series_key = time_series_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.video_frames = video_frames
        self.worker_id = 0

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("json", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("json", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)
        self.data_dir = os.path.dirname(data_path)

        # Create label mapping
        self.label_vocab = self._create_label_vocab()
        print(f"Label vocab has size {self.label_vocab.__len__()}")

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

    def _create_label_vocab(self):
        """
        Create a mapping between string labels and indices.
        Returns a dictionary mapping labels to indices.
        """
        # Extract all unique labels
        all_labels = set()
        for row in self.dataset:
            if self.answer_key in row:
                all_labels.add(row[self.answer_key])

        # Sort labels alphabetically for consistency
        sorted_labels = sorted(list(all_labels))

        # Create mapping
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        logger.info(f"Created label vocabulary with {len(label_to_idx)} classes")

        return label_to_idx

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []

            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            if self.time_series_key in example and example[self.time_series_key]:
                content_list.append({"type": "time-series"})  # add time series token

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
                len(processing_class.apply_chat_template(messages,
                                                         add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = copy.deepcopy(self.dataset[index])
        prompt_str: str = row_dict[self.prompt_key]

        # Add label index for classification task
        label_str = row_dict[self.answer_key]
        if label_str in self.label_vocab:
            row_dict["label_idx"] = self.label_vocab[label_str]
        else:
            # Handle unseen labels
            logger.warning(f"Worker {self.worker_id}: Unknown label {label_str} for item {index}")
            row_dict["label_idx"] = -1  # Use -1 to indicate unknown label

        processed_images = []
        original_dimensions = []  # Store original image dimensions
        processed_time_series = []

        # Extract data_source and dataset

        # Set vision_path to a nonempty vision path
        # Or empty if both vision paths are empty
        vision_path = row_dict['images'] if len(row_dict['images']) != 0 else row_dict.get('videos', [])
        ts_path = row_dict.get('time-series', [])
        if ts_path is None:
            ts_path = []


        if 'How long will the patient stay in the hospital?' in prompt_str:
            row_dict["data_source"] = "multimodal"
            row_dict["dataset"] = "los_prediction"
        elif 'Will the patient survive for at least 48 hours?' in prompt_str:
            row_dict["data_source"] = "multimodal"
            row_dict["dataset"] = "48_ihm"
        elif len(vision_path) != 0:
            vision_path = vision_path[0]
            row_dict["data_source"] = vision_path.split("/")[0]
            row_dict["dataset"] = vision_path.split("/")[1]
        elif ts_path and len(ts_path) != 0:
            row_dict["data_source"] = "ecg"
            # dataset already set in json
        else:
            raise ValueError("No modality found.")

        if len(vision_path) == 0 and len(ts_path) > 0:
            vision_path = ts_path
        row_dict['vision_path'] = vision_path

        if self.image_key in row_dict and row_dict["images"]:
            for i, image_item in enumerate(row_dict["images"]):
                try:
                    logger.debug(f"Worker {self.worker_id}: Processing image {i} for item {index}")

                    if isinstance(image_item, str):
                        # Load the image if it's a path
                        full_path = os.path.join(self.data_dir, image_item)
                        logger.debug(f"Worker {self.worker_id}: Loading image {i} from {full_path}")

                        if not os.path.exists(full_path):
                            logger.warning(f"Worker {self.worker_id}: Image file not found: {full_path}")
                            image = Image.new("RGB", (224, 224), (255, 255, 255))
                        else:
                            image = Image.open(full_path)
                    else:
                        image = image_item

                    original_dimensions.append((image.width, image.height))
                    # Process the image
                    processed_images.append(self.process_image(image))

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing image {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())
                    original_dimensions.append((224, 224))

        # Process videos if they exist
        if "videos" in row_dict and row_dict["videos"]:
            logger.debug(f"Worker {self.worker_id}: Processing videos for item {index}")
            for i, video_item in enumerate(row_dict["videos"]):
                try:
                    if isinstance(video_item, str):
                        # Load the video if it's a path
                        full_path = os.path.join(self.data_dir, video_item)
                        logger.debug(f"Worker {self.worker_id}: Loading video {i} from {full_path}")

                        if not os.path.exists(full_path):
                            logger.warning(f"Worker {self.worker_id}: Video file not found: {full_path}")
                            # Add placeholder frames
                            for _ in range(self.video_frames):
                                processed_images.append(Image.new("RGB", (224, 224), (255, 255, 255)))
                                original_dimensions.append((224, 224))  # Add placeholder dimensions
                        else:
                            # Extract frames from video
                            video_frames = extract_video_frames(full_path, self.video_frames)
                            for frame in video_frames:
                                # Store original dimensions
                                original_dimensions.append((frame.width, frame.height))
                                processed_images.append(self.process_image(frame))
                    else:
                        logger.warning(
                            f"Worker {self.worker_id}: Video item type not supported: {type(video_item)}")

                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error processing video {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())

        if self.time_series_key in row_dict and row_dict[self.time_series_key]:
            logger.debug(f"Worker {self.worker_id}: Processing time series for item {index}")
            for i, time_series_item in enumerate(row_dict[self.time_series_key]):
                try:
                    if isinstance(time_series_item, str):
                        full_path = os.path.join(self.data_dir, time_series_item)
                        logger.debug(f"Worker {self.worker_id}: Loading time series {i} from {full_path}")

                        if not os.path.exists(full_path):
                            logger.warning(f"Worker {self.worker_id}: Time series file not found: {full_path}")
                            raise FileNotFoundError(f"Time series file not found: {full_path}")
                        else:
                            # Load the time series data
                            time_series = torch.load(full_path).to(torch.float32)
                            # if time_series.dtype == torch.bfloat16:
                            #     time_series = time_series.to(torch.float32)
                    else:
                        time_series = time_series_item
                    processed_time_series.append(time_series)

                except Exception as e:
                    logger.error(
                        f"Worker {self.worker_id}: Error processing time series {i} for item {index}: {str(e)}")
                    logger.error(traceback.format_exc())
                    time_series = torch.zeros((8, 2500), dtype=torch.float32)
                    processed_time_series.append(time_series)
        else:
            time_series = torch.zeros((8, 2500), dtype=torch.float32)
            processed_time_series.append(time_series)

        # get size from processed_images
        if len(processed_images) > 0:
            image_size = processed_images[0].size
            logger.debug(f"Worker {self.worker_id}: Processed images size: {image_size}")
        else:
            image_size = (224, 224)

        if len(processed_time_series) > 0:
            time_series_size = processed_time_series[0].size()
            logger.debug(f"Worker {self.worker_id}: Processed time series size: {time_series_size}")
        else:
            time_series_size = (8, 2500)

        # Load segmentation mask if available
        if "segmentation_path" in row_dict and row_dict["segmentation_path"]:
            try:
                seg_path = os.path.join(self.data_dir, row_dict["segmentation_path"])
                if os.path.exists(seg_path):
                    logger.debug(f"Worker {self.worker_id}: Loading segmentation mask from {seg_path}")
                    segmentation_mask = Image.open(seg_path)
                    # Process the segmentation mask if needed
                    row_dict["segmentation_mask"] = segmentation_mask
                else:
                    logger.warning(f"Worker {self.worker_id}: Segmentation mask not found: {seg_path}")
                    row_dict["segmentation_mask"] = None
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error loading segmentation mask: {str(e)}")
                logger.error(traceback.format_exc())
                row_dict["segmentation_mask"] = None
        else:
            row_dict["segmentation_mask"] = None

        # Ensure we have at least one image/frame
        if not processed_images:
            logger.debug(f"Worker {self.worker_id}: No images or videos found for item {index}, using placeholder")
            processed_images = [Image.new("RGB", (224, 224), (255, 255, 255))]
            original_dimensions = [(224, 224)]  # Add placeholder dimensions

        row_dict["images"] = processed_images
        row_dict["multi_modal_data"] = {
            "image": processed_images,
        }
        if processed_time_series:
            row_dict[self.time_series_key] = processed_time_series
            row_dict["multi_modal_data"][self.time_series_key] = processed_time_series

        # Replace all image tokens in prompt with placeholders
        prompt_str = prompt_str.replace("<video>", "<image>")
        if "<image>" not in prompt_str:
            prompt_str = "<image> " + prompt_str
        image_count_in_prompt = prompt_str.count("<image>")
        image_count = len(processed_images)
        if len(processed_images) > 1 and image_count_in_prompt < len(processed_images):
            # add more image tokens to prompt
            missing_count = len(processed_images) - image_count_in_prompt
            prompt_str = prompt_str.replace("<image>", "<image> " * (missing_count + 1), 1)
        image_count_in_prompt = prompt_str.count("<image>")
        assert image_count == image_count_in_prompt, f"Image count mismatch: {image_count} != {image_count_in_prompt}"
        row_dict[self.prompt_key] = prompt_str
        messages = self._build_messages(row_dict)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        try:
            kwargs = dict()
            if self.time_series_key in row_dict and row_dict[self.time_series_key]:
                kwargs["time_series_data"] = row_dict["multi_modal_data"][self.time_series_key]
            model_inputs = self.processor(
                images=row_dict["multi_modal_data"]["image"],
                text=[prompt],
                return_tensors="pt",
                **kwargs
            )

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing model inputs: {str(e)}")
            # remove image
            row_dict["images"] = [Image.new("RGB", (224, 224), (255, 255, 255)) for _ in range(image_count)]
            row_dict["multi_modal_data"]["image"] = row_dict["images"]
            kwargs = dict()
            if self.time_series_key in row_dict and row_dict[self.time_series_key]:
                kwargs["time_series_data"] = row_dict["multi_modal_data"][self.time_series_key]
            model_inputs = self.processor(
                images=row_dict["multi_modal_data"]["image"],
                text=[prompt],
                return_tensors="pt",
                **kwargs
            )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        # Resize segmentation mask to match the image dimensions in model_inputs
        if row_dict["segmentation_mask"] is not None:
            try:
                # Extract dimensions from image_grid_thw (time, height, width)
                # We need the height and width for resizing
                target_height, target_width = image_size

                logger.debug(f"Worker {self.worker_id}: Resizing segmentation mask to {target_width}x{target_height}")

                # Resize the segmentation mask to match the processed image dimensions
                resized_mask = row_dict["segmentation_mask"].resize(
                    (target_width, target_height),
                    resample=Image.Resampling.NEAREST
                )

                mask_array = np.array(resized_mask)

                # If mask is grayscale, add channel dimension
                if len(mask_array.shape) == 2:
                    mask_array = mask_array[np.newaxis, :, :]
                # If mask is RGB but we only need one channel for segmentation
                elif len(mask_array.shape) == 3 and mask_array.shape[2] == 3:
                    mask_array = np.mean(mask_array, axis=2)[np.newaxis, :, :]

                # Convert to torch tensor
                # mask_tensor = torch.from_numpy(mask_array).float()
                row_dict["segmentation_mask"] = mask_array

                logger.debug(f"Worker {self.worker_id}: Successfully resized segmentation mask")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error resizing segmentation mask: {str(e)}")
                logger.error(traceback.format_exc())

        if "segmentation_mask" not in row_dict or row_dict["segmentation_mask"] is None:
            target_width, target_height = image_size
            # row_dict["segmentation_mask"] = torch.zeros(1, target_height, target_width, dtype=torch.float32)
            row_dict["segmentation_mask"] = np.zeros((target_height, target_width), dtype=np.uint8)

        # Handle bounding box information
        if "bbox" in row_dict and row_dict["bbox"]:
            try:
                target_width, target_height = image_size

                # Get original dimensions of the corresponding image
                # We assume the bbox corresponds to the first image
                if original_dimensions:
                    original_width, original_height = original_dimensions[0]
                    # Resize the bounding box
                    resized_bbox = resize_bbox(
                        row_dict["bbox"],
                        original_width,
                        original_height,
                        target_width,
                        target_height
                    )

                    logger.debug(f"Worker {self.worker_id}: Resized bbox from {row_dict['bbox']} to {resized_bbox}. "
                                 f"Original dimensions: {original_dimensions[0]}, "
                                 f"Target dimensions: {target_width}x{target_height}")
                    row_dict["bbox"] = resized_bbox
                else:
                    logger.warning(f"Worker {self.worker_id}: No original dimensions available for bbox resizing")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error resizing bounding box: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            # Use empty list as placeholder if not available
            row_dict["bbox"] = [0, 0, 0, 0]

        # Make bbox tensor
        row_dict["bbox"] = torch.tensor(row_dict["bbox"], dtype=torch.float32)

        row_dict["multi_modal_inputs"] = dict(model_inputs)
        if self.processor is not None:
            # and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor"
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["ground_truth"] = row_dict[self.answer_key]
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        row_dict.pop("segmentation_path", None)
        return row_dict