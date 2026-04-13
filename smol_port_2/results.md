# SmolVLA Port Results
## BF16 Run Results:

  [neuron vs cpu]  max=5.32520  mean=0.18508  allclose=FAIL  
  first_action (neuron): tensor([[-0.2334,  0.1553,  0.6094,  2.2500,  0.1108,  0.2158]])  

Benchmark…  
  [neuron]  avg=71.4ms  min=71.4ms  max=71.5ms  (n=20)  

## F32 Run Results:  

  [neuron vs cpu]  max=7.70830  mean=0.26117  allclose=FAIL  
  first_action (neuron): tensor([[-0.2369,  0.1552,  0.6034,  2.2457,  0.1110,  0.2147]])  

Benchmark…  
  [neuron]  avg=73.5ms  min=73.5ms  max=73.6ms  (n=20)  

## Uniform BF16 Results:
  [neuron vs cpu]  max=6.29395  mean=0.22710  allclose=FAIL  
  first_action (neuron): tensor([[-0.2334,  0.1553,  0.6094,  2.2500,  0.1108,  0.2158]])  

Benchmark…  
  [neuron]  avg=71.4ms  min=71.3ms  max=71.6ms  (n=20)  


## Notes:
- Overall the port looks successful, output looks correct, working fast, consistent  
- Low MFU and HFU (12 and 20 percent), so running the model does not use a lot of the chip (most likely due to model size)  
- Latency while faster now is still slower than estimations for other GPUs and units
- However considering the cost advantage and when parallelizing cores this could still give advantages.
