import logging
import os
import pandas as pd

logger = logging.getLogger("sample_manager")


class VideoSampleManager:

    def __init__(
        self,
        sample_src_dir: str = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "src_videos"
        ),
        target_dir: str = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "preprocessed"
        ),
    ):
        self.sample_src_dir = sample_src_dir
        self.target_dir = target_dir
        self.samples = dict()
        self.read_samples()
        logger.info("Samples:" + str(self.samples))

    @staticmethod
    def _normalize_filepath(filepath: str):
        return os.path.abspath(filepath)

    def read_samples(self):
        for item in os.scandir(self.sample_src_dir):
            if not item.is_file():
                continue
            normalized_path = self._normalize_filepath(item.path)
            if not os.path.splitext(normalized_path)[1] == ".mp4":
                logger.debug(f"Skipped {normalized_path}")
                continue
            sample_id = os.path.basename(normalized_path)
            self.samples[sample_id] = {
                "id": sample_id,
                "path": normalized_path,
                "label": None,
            }

    def get_sample(self, sample_id: str):
        return self.samples[sample_id]

    def get_samples(self):
        return self.samples

    def set_label(self, sample_id: str, label: str):
        self.samples[sample_id]["label"] = label

    def set_ready(self, sample_id: str):
        path = self.get_sample(sample_id)["path"]
        os.rename(path, os.path.join(self.target_dir, os.path.basename(path)))
        self.write_sample_label(sample_id)
        del self.samples[sample_id]

    def write_sample_label(self, sample_id: str):
        logger.info("Done labeling samples")
        labels_csv_path = os.path.join(self.target_dir, "labels.csv")
        labels_df = pd.read_csv(labels_csv_path)
        sample = self.samples[sample_id]
        labels_df.loc[len(labels_df)] = [sample["path"], sample["label"]]
        labels_df = labels_df.drop_duplicates("filepath", keep="last")
        labels_df.to_csv(labels_csv_path, index=False)
        logger.info("labels.csv updated")
