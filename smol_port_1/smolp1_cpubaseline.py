import torch
import psutil
import time
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model_id = "lerobot/smolvla_base"
device = torch.device("cuda")

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
        print(f"{k}: {tuple(v.shape)}")

process = psutil.Process(os.getpid())
print(f"\nRAM MB before inference: {process.memory_info().rss / 1e6:.1f}")

with torch.inference_mode():
    for _ in range(3):
        _ = policy.select_action(batch)

times = []
with torch.inference_mode():
    for _ in range(20):
        t0 = time.time()
        out = policy.select_action(batch)
        times.append(time.time() - t0)

with torch.inference_mode():
    ref_out = policy.select_action(batch)

print("\nOutput:")
print("type:", type(ref_out))
if isinstance(ref_out, torch.Tensor):
    print("shape:", tuple(ref_out.shape))
    print("values:", ref_out)

print("\nLatency:")
print(f"avg: {sum(times)/len(times)*1000:.2f} ms")
print(f"min: {min(times)*1000:.2f} ms")
print(f"max: {max(times)*1000:.2f} ms")

print(f"\nRAM MB after inference: {process.memory_info().rss / 1e6:.1f}")