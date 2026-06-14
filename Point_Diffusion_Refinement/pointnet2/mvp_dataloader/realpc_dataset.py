import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
import os
import copy
import warnings
import h5py

# RealPC categories (21 total, 4 superclasses)
REALPC_CATEGORIES = [
    'chine_0', 'chine_1', 'chine_2', 'chine_3',
    'dutch_0', 'dutch_1', 'dutch_2', 'dutch_3', 'dutch_4',
    'hung_0', 'hung_1', 'hung_2', 'hung_3', 'hung_4', 'hung_5', 'hung_6', 'hung_7',
    'sncf_0', 'sncf_1', 'sncf_2', 'sncf_3',
]

CATEGORY_TO_LABEL = {cat: i for i, cat in enumerate(REALPC_CATEGORIES)}


def load_pcd(path):
    """Load a .pcd file and return an (N, 3) numpy array."""
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    return points


def resample_points(points, target_n, rng=None):
    """Resample a point cloud to exactly target_n points."""
    n = points.shape[0]
    if n == target_n:
        return points
    if rng is None:
        rng = np.random.default_rng()
    if n > target_n:
        idx = rng.choice(n, target_n, replace=False)
    else:
        idx = np.concatenate([
            np.arange(n),
            rng.choice(n, target_n - n, replace=True)
        ])
    return points[idx]


def normalize_pair(partial, complete):
    """Normalize partial and complete point clouds."""
    centroid = np.mean(complete, axis=0, keepdims=True)
    partial_centered = partial - centroid
    complete_centered = complete - centroid
    max_dist = np.max(np.linalg.norm(complete_centered, axis=1))
    if max_dist < 1e-8:
        max_dist = 1.0
    partial_norm = partial_centered / max_dist
    complete_norm = complete_centered / max_dist
    return partial_norm.astype(np.float32), complete_norm.astype(np.float32)


class RealPCDataset(data.Dataset):
    """RealPC dataset loader for PDR.

    Compatible with the ShapeNetH5 interface used by PDR.
    Returns dict with keys: 'partial', 'complete', 'label',
    and optionally 'generated' and 'XT'.
    """

    def __init__(self, data_dir, train=True, n_partial=2048, n_complete=2048,
                 num_scans_per_object=None, scale=1.0,
                 augmentation=False, return_augmentation_params=False,
                 include_generated_samples=False, generated_sample_path=None,
                 randomly_select_generated_samples=False,
                 use_mirrored_partial_input=False, number_partial_points=2048,
                 load_pre_computed_XT=False, T_step=100, XT_folder=None,
                 rank=0, world_size=1, append_samples_to_last_rank=True,
                 random_subsample=False, num_samples=1000,
                 val_split=0.0, val_mode=False, val_seed=42,
                 scan_seed=42,
                 novel_input=True, novel_input_only=False,
                 ):

        self.n_partial = n_partial
        self.n_complete = n_complete
        self.scale = scale
        self.train = train
        self.augmentation = augmentation
        self.return_augmentation_params = return_augmentation_params
        self.num_scans_per_object = num_scans_per_object
        self.scan_seed = scan_seed

        self.include_generated_samples = include_generated_samples
        self.load_pre_computed_XT = load_pre_computed_XT
        self.use_mirrored_partial_input = False
        self.random_subsample = False

        split = 'train' if train else 'test'
        complete_dir = os.path.join(data_dir, split, 'complete')
        partial_dir = os.path.join(data_dir, split, 'partial')

        # Discover all (category, model_id) pairs and their partial scans
        self.samples = []
        self.objects = []

        for cat in sorted(os.listdir(complete_dir)):
            if cat not in CATEGORY_TO_LABEL:
                continue
            label = CATEGORY_TO_LABEL[cat]
            cat_complete_dir = os.path.join(complete_dir, cat)
            cat_partial_dir = os.path.join(partial_dir, cat)

            for model_file in sorted(os.listdir(cat_complete_dir)):
                if not model_file.endswith('.pcd'):
                    continue
                model_id = model_file.replace('.pcd', '')
                complete_path = os.path.join(cat_complete_dir, model_file)
                partial_model_dir = os.path.join(cat_partial_dir, model_id)

                if not os.path.isdir(partial_model_dir):
                    warnings.warn(f"No partial scans for {cat}/{model_id}, skipping")
                    continue

                partial_files = sorted([
                    os.path.join(partial_model_dir, f)
                    for f in os.listdir(partial_model_dir)
                    if f.endswith('.pcd')
                ])

                if len(partial_files) == 0:
                    warnings.warn(f"No .pcd files in {partial_model_dir}, skipping")
                    continue

                self.objects.append((complete_path, partial_files, label))

        # Optional: split train into train/val
        if train and val_split > 0:
            rng = np.random.default_rng(val_seed)
            n_objects = len(self.objects)
            n_val = max(1, int(n_objects * val_split))
            perm = rng.permutation(n_objects)
            val_idx = set(perm[:n_val].tolist())
            train_idx = set(perm[n_val:].tolist())
            if val_mode:
                self.objects = [self.objects[i] for i in sorted(val_idx)]
            else:
                self.objects = [self.objects[i] for i in sorted(train_idx)]

        # Build flat sample list with fixed seed for deterministic ordering
        self._build_sample_list()

        # Load pre-computed XT (intermediate diffusion results)
        self.generated_XT = None
        if load_pre_computed_XT:
            xt_split = 'train' if train else 'test'
            XT_file = os.path.join(XT_folder, xt_split,
                                   'realpc_generated_data_%dpts_T%d.h5' % (n_complete, T_step))
            print(f"Loading pre-computed XT from {XT_file}")
            with h5py.File(XT_file, 'r') as f:
                self.generated_XT = np.array(f['data'])
            self.generated_XT = self.generated_XT * 2 * scale
            print(f"Loaded XT: {self.generated_XT.shape}")
            assert self.generated_XT.shape[0] == len(self.samples), \
                f"XT samples ({self.generated_XT.shape[0]}) != dataset samples ({len(self.samples)})"

        # Load generated samples (CGNet completions for RFNet training)
        self.generated_sample = None
        if include_generated_samples:
            gen_split = 'train' if train else 'test'
            if randomly_select_generated_samples:
                gen_base = generated_sample_path
                trial_dirs = [d for d in os.listdir(gen_base) if d.startswith('trial')]
                if trial_dirs:
                    import random
                    selected_trial = random.choice(trial_dirs)
                    gen_file = os.path.join(gen_base, selected_trial, gen_split,
                                           'realpc_generated_data_%dpts.h5' % n_complete)
                    print(f"Randomly selected trial: {selected_trial}")
                else:
                    gen_file = os.path.join(gen_base, gen_split,
                                           'realpc_generated_data_%dpts.h5' % n_complete)
            else:
                gen_file = os.path.join(generated_sample_path, gen_split,
                                       'realpc_generated_data_%dpts.h5' % n_complete)
            print(f"Loading generated samples from {gen_file}")
            with h5py.File(gen_file, 'r') as f:
                self.generated_sample = np.array(f['data'])
            self.generated_sample = self.generated_sample * 2 * scale
            print(f"Loaded generated samples: {self.generated_sample.shape}")
            assert self.generated_sample.shape[0] == len(self.samples), \
                f"Generated samples ({self.generated_sample.shape[0]}) != dataset samples ({len(self.samples)})"

        print(f"RealPC [{split}{'_val' if val_mode else ''}]: "
              f"{len(self.objects)} objects, {len(self.samples)} samples, "
              f"partial={n_partial}pts, complete={n_complete}pts")

    def _build_sample_list(self):
        """Flatten objects into individual (partial, complete, label) samples.
        Uses a fixed seed for deterministic ordering across runs."""
        self.samples = []
        rng = np.random.default_rng(self.scan_seed)
        for complete_path, partial_files, label in self.objects:
            if self.num_scans_per_object is not None and self.num_scans_per_object < len(partial_files):
                selected = rng.choice(len(partial_files), self.num_scans_per_object, replace=False)
                selected_files = [partial_files[i] for i in selected]
            else:
                selected_files = partial_files
            for pf in selected_files:
                self.samples.append((pf, complete_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        partial_path, complete_path, label = self.samples[index]

        # Load point clouds
        partial_pts = load_pcd(partial_path)
        complete_pts = load_pcd(complete_path)

        # Resample to fixed size
        rng = np.random.default_rng()
        partial_pts = resample_points(partial_pts, self.n_partial, rng)
        complete_pts = resample_points(complete_pts, self.n_complete, rng)

        # Normalize (center on GT centroid, scale to unit sphere)
        partial_pts, complete_pts = normalize_pair(partial_pts, complete_pts)

        # Scale to match PDR range ([-scale, scale])
        partial_pts = partial_pts * self.scale
        complete_pts = complete_pts * self.scale

        # Augmentation
        if isinstance(self.augmentation, dict):
            try:
                from mvp_dataloader.mvp_data_utils import augment_cloud
                result_list = [partial_pts, complete_pts]
                if self.return_augmentation_params:
                    result_list, aug_params = augment_cloud(
                        result_list, self.augmentation, return_augmentation_params=True)
                else:
                    result_list = augment_cloud(
                        result_list, self.augmentation, return_augmentation_params=False)
                partial_pts, complete_pts = result_list[0], result_list[1]
            except ImportError:
                pass

        result = {
            'partial': torch.from_numpy(partial_pts).float(),
            'complete': torch.from_numpy(complete_pts).float(),
            'label': label,
        }

        # Load generated samples (CGNet completions)
        if self.include_generated_samples and self.generated_sample is not None:
            gen = copy.deepcopy(self.generated_sample[index])
            if isinstance(self.augmentation, dict):
                sigma = self.augmentation.get('noise_magnitude_for_generated_samples', 0)
                if sigma > 0:
                    noise = np.random.normal(scale=sigma, size=gen.shape).astype(gen.dtype)
                    gen = gen + noise
            result['generated'] = torch.from_numpy(gen).float()

        # Load pre-computed XT
        if self.load_pre_computed_XT and self.generated_XT is not None:
            result['XT'] = torch.from_numpy(
                copy.deepcopy(self.generated_XT[index])).float()

        return result


if __name__ == '__main__':
    if __name__ == '__main__':
        dataset = RealPCDataset(
            data_dir='/home/tepper/generative-point-cloud-completion/realpc/difficult',
            train=True,
            n_partial=2048,
            n_complete=5000,
            num_scans_per_object=26,
            scale=1.0,
        )

        loader = data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        for i, batch in enumerate(loader):
            print(f"Batch {i}: partial {batch['partial'].shape} "
                  f"[{batch['partial'].min():.3f}, {batch['partial'].max():.3f}], "
                  f"complete {batch['complete'].shape} "
                  f"[{batch['complete'].min():.3f}, {batch['complete'].max():.3f}], "
                  f"labels {batch['label']}")
            if i >= 2:
                break