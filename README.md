# Trainium Model Translation

This repository contains code for translating pytorch models onto trainium using Claude Code Skills. I am building off of Kevin Gomes' LLM and VLM NxDI porting skill to add support to VLAs through new skills and expanding NxDI primitives to the action head across denoising and discrete token action heads. 

## The skills

The skill is centered around SKILL_Port. This is the overall skill which takes in the input and manages everything. It first dispatches an agent to explore the ecosystem and find information on the model. Then it gives this information to SKILL and translates the model. There is a bunch of other infrastructure in their to expand the NxDI infrastructure and provide the skill with special guidance for action heads. 

## Next steps

I would like to do a final run through to verify this before it is used further and packaged. I am thinking I will run it on pi0. In addition exploring the applicability of this to trn2/3 since all ports have been on trn1.32xlarge.

## Software Log

### Smol Ports 1 and 2

These ports were manual and CC led ports before I had the skill. These are not that useful and were just for me to get used to the trainium ecosystem.

### smol_git and qwen3-port-git

These ports were my first using the CC skill. Also not that useful, as it is mostly unverified and just was used to find holes in the skill for action heads

### smol-skilled-port

This port used the infrastructure and skills set up for the action head producing better results but some issues with defaulting to tracing

### groot-n1-port-github

Full pretty much finished runthrough of the port with the skills and infrastructure, produced correct and solid results from model to output with guide on how to use the model.
