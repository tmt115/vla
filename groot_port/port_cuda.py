import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, ProcessorMixin

try:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.interfaces import BaseProcessor
    from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType
    from gr00t.data.types import MessageType, ModalityConfig, VLAStepData
    from gr00t.policy.policy import BasePolicy, PolicyWrapper
    _GROOT_INSTALLED = True
except ImportError:
    _GROOT_INSTALLED = False

    class EmbodimentTag(Enum):
        ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
        GR1 = "gr1"
        UNITREE_G1 = "unitree_g1"
        LIBERO_PANDA = "libero_panda"
        OXE_GOOGLE = "oxe_google"
        OXE_WIDOWX = "oxe_widowx"
        OXE_DROID = "oxe_droid"
        BEHAVIOR_R1_PRO = "behavior_r1_pro"
        NEW_EMBODIMENT = "new_embodiment"

    class MessageType(Enum):
        START_OF_EPISODE = "start_of_episode"
        END_OF_EPISODE = "end_of_episode"
        EPISODE_STEP = "episode_step"
        IMAGE = "image"
        TEXT = "text"

    class ActionRepresentation(Enum):
        RELATIVE = "relative"
        DELTA = "delta"
        ABSOLUTE = "absolute"

    class ActionType(Enum):
        EEF = "eef"
        NON_EEF = "non_eef"

    class ActionFormat(Enum):
        DEFAULT = "default"
        XYZ_ROT6D = "xyz+rot6d"
        XYZ_ROTVEC = "xyz+rotvec"

    @dataclass
    class VLAStepData:
        images: dict[str, list[np.ndarray]]
        states: dict[str, np.ndarray]
        actions: dict[str, np.ndarray]
        masks: dict[str, list[np.ndarray]] | None = None
        text: str | None = None
        embodiment: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
        is_demonstration: bool = False
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class ActionConfig:
        rep: ActionRepresentation
        type: ActionType
        format: ActionFormat
        state_key: str | None = None

    @dataclass
    class ModalityConfig:
        delta_indices: list[int]
        modality_keys: list[str]
        sin_cos_embedding_keys: list[str] | None = None
        mean_std_embedding_keys: list[str] | None = None
        action_configs: list[ActionConfig] | None = None

        def __post_init__(self):
            if self.action_configs is not None:
                assert len(self.action_configs) == len(self.modality_keys)
                parsed = []
                for ac in self.action_configs:
                    if isinstance(ac, dict):
                        ac = ActionConfig(
                            rep=ActionRepresentation[ac["rep"]],
                            type=ActionType[ac["type"]],
                            format=ActionFormat[ac["format"]],
                            state_key=ac.get("state_key"),
                        )
                    parsed.append(ac)
                self.action_configs = parsed

    class BaseProcessor(ProcessorMixin):
        def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
            raise NotImplementedError

        def decode_action(
            self,
            action: np.ndarray,
            embodiment_tag: EmbodimentTag,
            state: dict[str, np.ndarray] | None = None,
        ) -> dict[str, np.ndarray]:
            raise NotImplementedError

        @property
        def collator(self):
            raise NotImplementedError

        @abstractmethod
        def set_statistics(self, statistics: dict[str, Any], override: bool = False) -> None:
            pass

        def train(self):
            self.training = True

        def eval(self):
            self.training = False

        def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
            return getattr(self, "modality_configs")

    class BasePolicy(ABC):
        def __init__(self, *, strict: bool = True):
            self.strict = strict

        @abstractmethod
        def check_observation(self, observation: dict[str, Any]) -> None:
            pass

        @abstractmethod
        def check_action(self, action: dict[str, Any]) -> None:
            pass

        @abstractmethod
        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            pass

        def get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            if self.strict:
                self.check_observation(observation)
            action, info = self._get_action(observation, options)
            if self.strict:
                self.check_action(action)
            return action, info

        @abstractmethod
        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            pass

    class PolicyWrapper(BasePolicy):
        def __init__(self, policy: BasePolicy, *, strict: bool = True):
            super().__init__(strict=strict)
            self.policy = policy

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            return self.policy.reset(options)


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    return x


class Gr00tPolicy(BasePolicy):
    def __init__(
        self,
        model_path: str,
        embodiment_tag: EmbodimentTag,
        *,
        device: int | str,
        strict: bool = True,
    ):
        if _GROOT_INSTALLED:
            import gr00t.model

        super().__init__(strict=strict)
        local = Path(model_path)
        model_dir = local if local.exists() else model_path

        model = AutoModel.from_pretrained(model_dir)
        model.eval()
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model
        self.device = device

        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[embodiment_tag.value]
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        batch_size = value["video"][next(iter(value["video"]))].shape[0]
        return [
            {
                "video":    {k: v[i] for k, v in value["video"].items()},
                "state":    {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            for i in range(batch_size)
        ]

    def _to_vla_step_data(self, obs: dict[str, Any]) -> VLAStepData:
        return VLAStepData(
            images=obs["video"],
            states=obs["state"],
            actions={},
            text=obs["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        for modality in ("video", "state", "language"):
            assert modality in observation, f"Observation missing '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"observation['{modality}'] must be a dict, got {type(observation[modality])}"
            )

        bs = -1

        for key in self.modality_configs["video"].modality_keys:
            assert key in observation["video"], f"Missing video key '{key}'"
            arr = observation["video"][key]
            bs = arr.shape[0] if bs == -1 else bs
            assert arr.shape[0] == bs, f"video['{key}'] batch size mismatch"
            assert isinstance(arr, np.ndarray), f"video['{key}'] must be np.ndarray"
            assert arr.dtype == np.uint8, f"video['{key}'] must be uint8"
            assert arr.ndim == 5, f"video['{key}'] must be (B,T,H,W,C), got {arr.shape}"
            assert arr.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"video['{key}'] temporal dim mismatch"
            )
            assert arr.shape[-1] == 3, f"video['{key}'] must have 3 channels"

        for key in self.modality_configs["state"].modality_keys:
            assert key in observation["state"], f"Missing state key '{key}'"
            arr = observation["state"][key]
            bs = arr.shape[0] if bs == -1 else bs
            assert arr.shape[0] == bs, f"state['{key}'] batch size mismatch"
            assert isinstance(arr, np.ndarray), f"state['{key}'] must be np.ndarray"
            assert arr.dtype == np.float32, f"state['{key}'] must be float32"
            assert arr.ndim == 3, f"state['{key}'] must be (B,T,D), got {arr.shape}"
            assert arr.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"state['{key}'] temporal dim mismatch"
            )

        for key in self.modality_configs["language"].modality_keys:
            assert key in observation["language"], f"Missing language key '{key}'"
            batched = observation["language"][key]
            bs = len(batched) if bs == -1 else bs
            assert len(batched) == bs, f"language['{key}'] batch size mismatch"
            assert isinstance(batched, list), f"language['{key}'] must be a list"
            for item in batched:
                assert isinstance(item, list), "language batch item must be a list"
                assert len(item) == 1, "language batch item must have exactly one string"
                assert isinstance(item[0], str), "language item must be a str"

    def check_action(self, action: dict[str, Any]) -> None:
        for key in self.modality_configs["action"].modality_keys:
            assert key in action, f"Missing action key '{key}'"
            arr = action[key]
            assert isinstance(arr, np.ndarray), f"action['{key}'] must be np.ndarray"
            assert arr.dtype == np.float32, f"action['{key}'] must be float32"
            assert arr.ndim == 3, f"action['{key}'] must be (B,T,D), got {arr.shape}"
            assert arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"action['{key}'] temporal dim mismatch"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        unbatched = self._unbatch_observation(observation)

        processed = []
        states = []
        for obs in unbatched:
            step_data = self._to_vla_step_data(obs)
            states.append(step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            processed.append(self.processor(messages))

        batch = self.collate_fn(processed)
        batch = _rec_to_dtype(batch, dtype=torch.bfloat16)

        with torch.inference_mode():
            pred = self.model.get_action(**batch)
        normalized_action = pred["action_pred"].float()

        batched_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in self.modality_configs["state"].modality_keys
        }
        unnormalized = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        return {k: v.astype(np.float32) for k, v in unnormalized.items()}, {}

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1

    def _resolve_language_key(self, observation: dict[str, Any], key: str) -> str:
        if key == "task" and "annotation.human.coarse_action" in observation:
            return "annotation.human.coarse_action"
        return key

    def check_observation(self, observation: dict[str, Any]) -> None:
        cfg = self.get_modality_config()

        for key in cfg["video"].modality_keys:
            flat = f"video.{key}"
            assert flat in observation, f"Missing '{flat}'"
            arr = observation[flat]
            assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
            assert arr.ndim == 5, f"'{flat}' must be (B,T,H,W,C)"
            assert arr.shape[1] == len(cfg["video"].delta_indices)
            assert arr.shape[-1] == 3

        for key in cfg["state"].modality_keys:
            flat = f"state.{key}"
            assert flat in observation, f"Missing '{flat}'"
            arr = observation[flat]
            assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
            assert arr.ndim == 3, f"'{flat}' must be (B,T,D)"
            assert arr.shape[1] == len(cfg["state"].delta_indices)

        for key in cfg["language"].modality_keys:
            resolved = self._resolve_language_key(observation, key)
            assert resolved in observation, f"Missing language key '{resolved}'"
            lang = observation[resolved]
            assert isinstance(lang, (tuple, list))
            assert isinstance(lang[0], str)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = self.policy.modality_configs
        new_obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

        for key in cfg["video"].modality_keys:
            new_obs["video"][key] = observation[f"video.{key}"]

        for key in cfg["state"].modality_keys:
            new_obs["state"][key] = observation[f"state.{key}"]

        for key in cfg["language"].modality_keys:
            resolved = self._resolve_language_key(observation, key)
            new_obs["language"][key] = [[str(item)] for item in observation[resolved]]

        action, info = self.policy.get_action(new_obs, options)
        return {f"action.{k}": v for k, v in action.items()}, info

    def check_action(self, action: dict[str, Any]) -> None:
        cfg = self.get_modality_config()
        for key in cfg["action"].modality_keys:
            flat = f"action.{key}"
            assert flat in action, f"Missing '{flat}'"
            arr = action[flat]
            assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
            assert arr.ndim == 3
            assert arr.shape[1] == len(cfg["action"].delta_indices)

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.policy.get_modality_config()


def benchmark_policy(
    policy: Gr00tPolicy,
    observation: dict[str, Any],
    n_warmup: int = 5,
    n_trials: int = 20,
) -> dict[str, Any]:
    import time

    device = policy.device
    use_cuda = isinstance(device, str) and device.startswith("cuda") or isinstance(device, int)

    def cuda_sync():
        if use_cuda:
            torch.cuda.synchronize()

    def now():
        cuda_sync()
        return time.perf_counter()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    pre_times, inf_times, post_times = [], [], []

    total_runs = n_warmup + n_trials
    for i in range(total_runs):
        t0 = now()
        unbatched = policy._unbatch_observation(observation)
        processed = []
        states = []
        for obs in unbatched:
            step_data = policy._to_vla_step_data(obs)
            states.append(step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            processed.append(policy.processor(messages))
        batch = policy.collate_fn(processed)
        batch = _rec_to_dtype(batch, dtype=torch.bfloat16)
        t1 = now()

        with torch.inference_mode():
            pred = policy.model.get_action(**batch)
        normalized_action = pred["action_pred"].float()
        t2 = now()

        batched_states = {
            k: np.stack([s[k] for s in states], axis=0)
            for k in policy.modality_configs["state"].modality_keys
        }
        policy.processor.decode_action(
            normalized_action.cpu().numpy(), policy.embodiment_tag, batched_states
        )
        t3 = now()

        if i >= n_warmup:
            pre_times.append((t1 - t0) * 1000)
            inf_times.append((t2 - t1) * 1000)
            post_times.append((t3 - t2) * 1000)

    def stats(times):
        arr = np.array(times)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    total_times = [p + i + q for p, i, q in zip(pre_times, inf_times, post_times)]
    peak_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else float("nan")
    )

    return {
        "preprocess_ms":  stats(pre_times),
        "inference_ms":   stats(inf_times),
        "postprocess_ms": stats(post_times),
        "total_ms":       stats(total_times),
        "throughput_hz":  1000.0 / np.mean(total_times),
        "peak_gpu_mb":    peak_mb,
        "n_warmup":       n_warmup,
        "n_trials":       n_trials,
    }


def print_benchmark_results(results: dict[str, Any]) -> None:
    w = 16
    print(f"\n{'=' * 52}")
    print(f"  GR00T Benchmark  ({results['n_trials']} trials, {results['n_warmup']} warmup)")
    print(f"{'=' * 52}")
    for stage in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
        s = results[stage]
        label = stage.replace("_ms", "").ljust(w)
        print(f"  {label}  {s['mean']:7.2f} ms  ±{s['std']:5.2f}  "
              f"[{s['min']:.2f}, {s['max']:.2f}]")
    print(f"  {'throughput'.ljust(w)}  {results['throughput_hz']:.2f} steps/sec")
    if not np.isnan(results["peak_gpu_mb"]):
        print(f"  {'peak GPU mem'.ljust(w)}  {results['peak_gpu_mb']:.1f} MB")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Benchmark GR00T policy inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--embodiment_tag", default="gr1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_h", type=int, default=224)
    parser.add_argument("--img_w", type=int, default=224)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    tag = EmbodimentTag(args.embodiment_tag)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=tag,
        device=args.device,
    )

    cfg = policy.modality_configs
    B = args.batch_size
    T_vid   = len(cfg["video"].delta_indices)
    T_state = len(cfg["state"].delta_indices)

    _sap_stats = policy.processor.state_action_processor.statistics
    _etag = tag.value
    def _state_dim(key):
        try:
            modality_stats = _sap_stats[_etag]["state"]
            if key in modality_stats:
                return len(next(iter(modality_stats[key].values())))
            return sum(len(next(iter(v.values()))) for v in modality_stats.values())
        except Exception:
            return 7

    observation = {
        "video": {
            k: np.zeros((B, T_vid, args.img_h, args.img_w, 3), dtype=np.uint8)
            for k in cfg["video"].modality_keys
        },
        "state": {
            k: np.zeros((B, T_state, _state_dim(k)), dtype=np.float32)
            for k in cfg["state"].modality_keys
        },
        "language": {
            k: [["pick up the object"]] * B
            for k in cfg["language"].modality_keys
        },
    }

    results = benchmark_policy(policy, observation, n_warmup=args.n_warmup, n_trials=args.n_trials)
    print_benchmark_results(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")
