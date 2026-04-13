# SmolVLA
## Baseline 1:
Tensor shapes:  
action: (1, 7)  
index: (1,)  
task_index: (1,)  
episode_index: ()  
observation.state: (1, 8)  
observation.images.camera1: (1, 3, 256, 256)  
observation.images.camera2: (1, 3, 256, 256)  
observation.images.camera3: (1, 3, 256, 256)  
observation.language.tokens: (1, 48)  
observation.language.attention_mask: (1, 48)  

RAM MB before inference: 1336.7  
RAM MB after inference: 1339.4  

Output:  
type: <class 'torch.Tensor'>  
shape: (1, 6)  
values: tensor([[ 0.1087, -0.0582,  0.2657,  1.2845, -0.0738,  6.1350]])  

Latency:  
avg: 3.53 ms  
min: 3.00 ms  
max: 5.01 ms  

Cuda  

RAM MB before inference: 1285.5  
RAM MB after inference: 2148.2  

Output:  
type: <class 'torch.Tensor'>  
shape: (1, 6)  
values: tensor([[-0.1968, -0.0688, -0.0090,  1.3006, -0.1120,  5.2340]],  device='cuda:0')  

Latency:  
avg: 3.43 ms  
min: 3.00 ms  
max: 6.00 ms  

## Baseline 2A: Port Modeling CPU
RAM MB before inference: 1875.2  
RAM MB after inference: 1915.1  

raw_out shape: (1, 50, 32)  
trimmed_out shape: (1, 50, 6)  
first_action shape: (1, 6)  
latency_ms: 19114.554166793823  
first_action: tensor([[-0.1177,  0.0596,  0.7512,  2.3409, -0.0414, -0.4326]])  

## Baseline 2B: Port Modelin CUDA
RAM MB before inference: 1324.0  
RAM MB after inference: 2186.6  

raw_out shape: (1, 50, 32)  
trimmed_out shape: (1, 50, 6)  
first_action shape: (1, 6)  
latency_ms: 1547.8339195251465  
first_action: tensor([[-0.1919,  0.1107,  0.7706,  2.3058, -0.0389, -0.0919]], device='cuda:0')  

## First Neuron Run:
cpu_raw_out shape: (1, 50, 32)  
cpu latency_ms: 1776.9114971160889  

Comparing saved_ref_raw_out vs trainium_box_cpu_raw_out  
saved_ref_raw_out shape: (1, 50, 32)  
trainium_box_cpu_raw_out shape: (1, 50, 32)  
max abs diff: 2.622800588607788  
mean abs diff: 0.09194841980934143  
allclose(atol=1e-3, rtol=1e-3): False  

Tracing with torch_neuronx...  
................................................  
Compiler status PASS  

=== Neuron output ===  
neuron_raw_out shape: (1, 50, 32)  
neuron latency_ms: 19133.94522666931  

Comparing trainium_box_cpu_raw_out vs neuron_raw_out  
trainium_box_cpu_raw_out shape: (1, 50, 32)  
neuron_raw_out shape: (1, 50, 32)  
max abs diff: 4.368988037109375  
mean abs diff: 0.15313975512981415  
allclose(atol=1e-3, rtol=1e-3): False  

=== Postprocessed comparison ===  
cpu_trimmed shape: (1, 50, 6)  
cpu_first shape: (1, 6)  
neuron_trimmed shape: (1, 50, 6)  
neuron_first shape: (1, 6)  
neuron_first: tensor([[-0.2369,  0.1552,  0.6034,  2.2457,  0.1110,  0.2147]])  

Comparing cpu_first vs neuron_first  
cpu_first shape: (1, 6)  
neuron_first shape: (1, 6)  
max abs diff: 0.31389766931533813  
mean abs diff: 0.10700394958257675  
allclose(atol=1e-3, rtol=1e-3): False  

