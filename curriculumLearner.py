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

    def score_sample(self, sample):
        traj_sample = [{
            "RGB_im": sample["RGB_im"][0],      # (384, 384, 3)
            "depth": sample["depth"][0],        # (384, 384, 1)
            "new_traj": sample["new_traj"],  # bool
            "rot": sample["rot"][0],            # (4,)
            "trans": sample["trans"][0],        # (3,)
        }]


        camera_input = sample["camera"]

        print(type(sample["RGB_im"]), sample["RGB_im"].shape)
        # Prediction
        pred = self.model.predict([traj_sample, camera_input], verbose=0)

        target = tf.expand_dims(sample['depth'], axis=0)
        loss = tf.reduce_mean(tf.square(target - pred["depth"])).numpy()
        return loss

    def score_all_samples_and_save(self):
        print("Scoring samples for curriculum...")
        if os.path.exists(self.score_path):
            print("Scores already computed")
            return

        for sample in self.dataset_list:
            print("RGB shape:", sample["RGB_im"].shape)
            print("Keys:", sample.keys())
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
