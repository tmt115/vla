import torch
import psutil
import time
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model_id = "lerobot/smolvla_base"
device = torch.device("cuda")
print(device)

policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

dataset = LeRobotDataset("lerobot/libero")
frame = dict(dataset[0])

# Remap image names
frame["observation.images.camera1"] = frame.pop("observation.images.image")
frame["observation.images.camera2"] = frame.pop("observation.images.image2")
frame["observation.images.camera3"] = frame["observation.images.camera2"]

# Add a language field if the sample only has "task"
if "observation.language" not in frame:
    frame["observation.language"] = frame.get("task", "do the task")

# Add batch dim for images
for k in [
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
]:
    if isinstance(frame[k], torch.Tensor) and frame[k].ndim == 3:
        frame[k] = frame[k].unsqueeze(0)

# Add batch dim for state
if isinstance(frame["observation.state"], torch.Tensor) and frame["observation.state"].ndim == 1:
    frame["observation.state"] = frame["observation.state"].unsqueeze(0)

batch = preprocess(frame)

print("\nTensor shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, tuple(v.shape), v.dtype)

process = psutil.Process(os.getpid())
print(f"\nRAM MB before inference: {process.memory_info().rss / 1e6:.1f}")

with torch.inference_mode():
    t0 = time.time()

    core_batch = policy._prepare_batch(batch)
    images, img_masks = policy.prepare_images(core_batch)
    state = policy.prepare_state(core_batch)
    lang_tokens = core_batch["observation.language.tokens"]
    lang_masks = core_batch["observation.language.attention_mask"]

    raw_out = policy.model.sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )

    original_action_dim = policy.config.action_feature.shape[0]
    trimmed_out = raw_out[:, :, :original_action_dim]
    first_action = trimmed_out[:, 0]

    t1 = time.time()

print("raw_out shape:", tuple(raw_out.shape))
print("trimmed_out shape:", tuple(trimmed_out.shape))
print("first_action shape:", tuple(first_action.shape))
print("latency_ms:", (t1 - t0) * 1000)
print("first_action:", first_action)


print(f"\nRAM MB after inference: {process.memory_info().rss / 1e6:.1f}")