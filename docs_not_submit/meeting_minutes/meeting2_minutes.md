# Meeting 2 - Minutes 

**MRP**
- Model: 
	- Structure: CNN architecture with basic CAM (potential to extend CAMs if needed)
	- Architecture (CNN) will be the same for the fully-supervised (FS) and weakly-supervised (WS) model 
	- Implement regularisation method (https://arxiv.org/abs/2401.11122) on architecture and potentially compare with and without regularisation
- Tasks 
	1. Find CNN architecture to use for the baseline model for both WS and FS model (Anastasia/Jessica/Tony)
		1. Decide on already existing architecture 
		2. Tune hyper-parameters to fit this specific problem
	2. Implement CAM (Chenge/Futian/Henry)
	3. Implement Regularisation (Jose/Lok)
	4. Design and Conduct Experiments (Chenge/Futian/Henry)
		1. Compare WS to FS 
		2. Ablation Study by varying important hyper- parameters 

**OEQ**
- TBD once MRP finalised 
- Idea: Compare above structure with transformer-driven weakly supervised segmentation (see Chenge's lit review for suggestions) with optional text prompts as auxiliary for the transformer 

**Next Steps**
- Each person assigned to task effectively in chronological order 
- Once MRP code has been finalised, schedule next meet-up for OEQ 

TBD:   
- *Formulate a specific weak supervision problem to address and justify its usefulness and feasibility.*
