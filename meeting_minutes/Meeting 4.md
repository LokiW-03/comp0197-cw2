# Meeting 4 Summary 

### Outstanding Tasks

**Everyone**
- Ensure no redundant code in the main branch for easy collation

**Henry and Jessica** 
- Run on test set 
- Merge Open-Ended branch into main
- Update readme to clearly state (according to instructions):
  - The hypothesis stated
  - Experiments designed and run
  - Results obtained for quantitative conclusion
  - Qualitative overview/significance of results (effectively 'Why was this an interesting open-ended question')
 
**Jessica** 
- Merge all code together into zip file along with pdf instructions so to submit

**Lok**
- Update ReadMe with correct accuracies
- Remove any references to ScoreCam as code didn't work in end
- Update ReadMe with explanation of why we aren't using the CRM for the "final" models (in comparison and ablation), including the chart shown in the call
 
**Anastasia**  

- Add bibliogrpahy 

**Chenge and Tony**
- Finish Running experiments
- Summarise results in a table and with relevant metrics charts
- ReadMe:
  - Explain what model has been chosen as the baseline (prior to grid search) for comparison purposes, outlining the hyperparameters involved and threshold values, and showing results and metrics chart. This will be a model similar to that from the FS scripts
  - Summarise what experiments have been run
  - Include visualizations discussed in call (montage, metrics graphs, heatmaps/pseudomasks) 

### Visualisation Overview  

- Lok added epoch vs loss for CRM and ResNet training which was part of the finetuning process. Mentioned this was less interesting for the report 
- Tony added metrics plot for the fully-supervised models as well as a montage
- Chenge added CAM threshold plot (ablation/results).  Mentioned this is interesting to view but less interesting for report. Note the ablation/viz folder is not relevatn to this project. 


### Emergency Meeting Timing Details 
If there is any issue, we will all meet again Saturday at 10am on zoom. 

### Other items discussed

**Comparison of FS and WSS**  

- Rather than choosing the best FS model and then using it in WS learning, we took all four models and investigated their performance in the weakly supervised setting, as strong performance in a fully supervised task does not guarantee strong performance in weakly supervised task.

**Fully-Supervised Models**
- Hyperparameter tuning was performed on the FS models, however this is less important to the report itself (according to instructions) and thus the we will only be including visualizations on the initial models.
- Hyperparameters as follows:
  - Epochs: 10
  - Batch size: 16
  - Optimizer: AdamW
  - LR 1e-3
  - Momentum: NA
  - Weight Decay: 1e-4
  - LR Scheduler: StepLR
  - Step_size = 15
  - gamma = 0.1
  - No data transform or dropout  




  

