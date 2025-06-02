import numpy as np
import os
import tensorflow as tf


class CurriculumLearnerM4DepthStep:
    def __init__(self, model, dataset_list, model_ckpt_path, pacing_function="linear", total_steps=10000, a=0.4, b=0.2, p=2, score_path="sample_scores.npy"):
        """
        model: Keras model with loaded weights
        dataset_list: list of individual sample dictionaries
        model_ckpt_path: path to model checkpoint
        pacing_function: 'linear' or 'quadratic'
        total_steps: total training steps
        """
        self.model = model
        self.dataset_list = dataset_list
        self.model_ckpt_path = model_ckpt_path
        self.pacing_function = pacing_function
        self.total_steps = total_steps
        self.a = a
        self.b = b
        self.p = p
        self.score_path = score_path

        self.total_samples = len(dataset_list)
        self.scored_dataset = []  # (score, sample_dict)
        self.sorted_samples = []
        self.load_weights()

    def load_weights(self):
        self.model.load_weights(self.model_ckpt_path)

    def save_scores(self, path="sample_scores.npy"):
        np.save(path, np.array([s[0] for s in self.scored_dataset]))

    def load_scores(self, path="sample_scores.npy"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Score file {path} not found. Run score_all_samples_and_save() first.")
        scores = np.load(path)
        self.sorted_samples = sorted(zip(scores, self.dataset_list), key=lambda x: x[0])

    def merge_sequence_to_input_dict(self, sequence):
        """
        Convert a list of frame dictionaries into a model-ready input dict
        by stacking values and skipping non-array entries (e.g., 'camera', 'new_traj', 'depth').
        """
        input_dict = {}
        for key in sequence[0].keys():
            if key in ["depth", "camera"]:
                continue  # Exclude non-input fields
            values = [frame[key] for frame in sequence]
            if isinstance(values[0], np.ndarray):
                input_dict[key] = np.stack(values, axis=0)  # shape: (T, ...)
        return input_dict

    def score_sample(self, sample):
        traj_sample = sample["traj"] #list of 4 dicts
        camera_input = sample["camera"]

        # Add batch dimension to all inputs
        model_input = self.merge_sequence_to_input_dict(traj_sample)

        # Add batch dimension
        model_input = {k: np.expand_dims(v, axis=0) for k, v in model_input.items()}
        camera_input = {k: np.expand_dims(v, axis=0) for k, v in camera_input.items()}

        # Merge into a single dict, as expected by predict_step()
        model_input["camera"] = camera_input


        print({k: v.shape for k, v in model_input.items()})
        pred = self.model.predict(model_input, verbose=0)

        target = np.expand_dims(traj_sample[-1]["depth"], axis=0)  # shape: (1, H, W, 1)
        loss = np.mean((target - pred["depth"]) ** 2)
        return loss

    def score_all_samples_and_save(self):
        print("Scoring samples for curriculum...")
        if os.path.exists(self.score_path):
            print("Scores already computed")
            return

        for sample in self.dataset_list:
            #print("RGB shape:", sample["RGB_im"].shape)
            #print("Keys:", sample.keys())
            score = self.score_sample(sample)
            self.scored_dataset.append((score, sample))
        self.sorted_samples = sorted(self.scored_dataset, key=lambda x: x[0])
        self.save_scores(self.score_path)

    def pacing(self, step):
        t = step + 1
        Nb = int(self.b * self.total_samples)
        aT = self.a * self.total_steps

        if self.pacing_function == "linear":
            return int(Nb + ((1 - self.b) * self.total_samples / aT) * t)
        elif self.pacing_function == "quadratic":
            return int(Nb + (self.total_samples * (1 - self.b) / (aT ** self.p)) * (t ** self.p))
        elif self.pacing_function == "exponential":
            return int(Nb + (self.total_samples * (1 - self.b) / (np.exp(10) - 1)) * (np.exp(10 * t / aT) - 1))
        elif self.pacing_function == "logarithmic":
            return int(Nb + self.total_samples * (1 - self.b) * (1 + (1 / 10) * np.log(t / aT + np.exp(-10))))
        elif self.pacing_function == "step":
            return int(Nb + self.total_samples * (0 if (t / aT) < 1 else 1))
        else:
            raise ValueError(f"Unknown pacing function: {self.pacing_function}")

    def get_dataset_for_step(self, step, batch_size=3):
        if not self.sorted_samples:
            self.load_scores(self.score_path)

        n = self.pacing(step)
        n = min(n, len(self.sorted_samples))
        selected = [s[1] for s in self.sorted_samples[:n]]

        dataset = tf.data.Dataset.from_generator(
            lambda: (sample for sample in selected),
            output_signature={
                'RGB_im': tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                'depth': tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
            }
        )
        return dataset.batch(batch_size)
