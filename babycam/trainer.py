from logging import getLogger
import os
import torch
import cv2
import albumentations as A
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo, Video
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo,
)

logger = getLogger()

NEW_CLIPS_DIR_PATH = "./new_clips"
TRAINING_CLIPS_DIR_PATH = "./training_clips"
N_OF_AUGMENTATIONS = 5
RESOLUTION = (640, 480)
FPS = 20.0

AUGMENTATION_TRANSFORMATIONS = A.ReplayCompose(
    [
        A.ElasticTransform(alpha=0.5, p=0.5),
        A.ShiftScaleRotate(scale_limit=0.05, rotate_limit=10, p=0.5),
        A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(p=0.5),
        A.PixelDropout(drop_value=0, dropout_prob=0.01, p=0.5),
        A.PixelDropout(drop_value=255, dropout_prob=0.01, p=0.5),
        A.Blur(blur_limit=(2, 4), p=0.5),
    ]
)


def load_video(video_path):
    frame_list = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        frame_list.append(frame)
    cap.release()

    video = EncodedVideo.from_path(video_path)
    return frame_list, len(frame_list), fps, int(video.duration)


def split_clips(video: Video, clip_length: float = 3.0):
    clip_starts_s = [
        x / 10.0 for x in range(0, int(video.duration * 10), int(clip_length * 10))
    ]
    return [video.get_clip(start + int(clip_length)) for start in clip_starts_s]


def augment_frames(frame_list):
    data = None
    augmented_frame_list = []

    for i, item in enumerate(frame_list):
        if i == 0:
            first_image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            new_image = AUGMENTATION_TRANSFORMATIONS(image=first_image)["image"]
        else:
            image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            new_image = AUGMENTATION_TRANSFORMATIONS.ReplayCompose.replay(
                data["replay"], image=image
            )["image"]
        augmented_frame_list.append(new_image)

    return augmented_frame_list


def write_augmented_clips(clip: Video, clip_name: str, n_of_augmentations: int):
    original_frames = clip.get_frames()
    output_frames = original_frames
    for i in range(n_of_augmentations + 1):
        if i > 0:  # First is the original
            output_frames = augment_frames(original_frames)
        output_clip_path = os.path.join(
            TRAINING_CLIPS_DIR_PATH,
            clip_name + "_c" + i + ".mp4",
        )
        cv2.VideoWriter(output_clip_path, output_frames, FPS, RESOLUTION)


def preprocess():
    for file in os.scandir(NEW_CLIPS_DIR_PATH):
        logger.debug(f"Processing: {file.path}")
        if not file.is_file:
            logger.debug(f"Skipped (not a file): {file.path}")
            continue
        video = EncodedVideo.from_path(file.path)
        if int(video.duration) == 0:
            logger.debug(f"Skipped (too short): {file.path}")
            continue

        clips = split_clips(video, 3)
        for i, clip in enumerate(clips):
            filename, _ = os.path.splitext(file.name)
            write_augmented_clips(clip, f"{filename}_f{i}", N_OF_AUGMENTATIONS)
        logger.debug(f"Done with: {file.path}")


def train():
    pass


def test():
    pass


def main():
    preprocess()
    train()
    test()


if __name__ == "__main__":
    main()
