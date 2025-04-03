# Weakly segmentation performance

!python -m model.train --model=segnet --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo

Best model found at epoch 2 with IoU **0.1466044932603836**
- Test -> {'loss': 3.3966317176818848, 'accuracy': 0.2635522782802582, 'precision': 0.2639669179916382, 'recall': 0.26904618740081787, 'iou': 0.1466044932603836, 'dice': 0.2552682161331177} 
- Train -> {'loss': 0.5243124965740287, 'accuracy': 0.773007530492285, 'precision': 0.7537428003290425, 'recall': 0.7438368932060574, 'iou': 0.6100058529687964, 'dice': 0.7464214786239293}

!python -m model.train --model=segnext --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo

Best model found at epoch 9 with IoU 0.1118076890707016
- Test -> {'loss': nan, 'accuracy': 0.3354230523109436, 'precision': 0.7784743309020996, 'recall': 0.3333333432674408, 'iou': 0.1118076890707016, 'dice': 0.16744910180568695} 
- Train -> {'loss': 0.26888966962047245, 'accuracy': 0.884371628968612, 'precision': 0.8725013302720112, 'recall': 0.8713583876257357, 'iou': 0.7779615656189297, 'dice': 0.8714245894680852}


!python -m model.train --model=effunet --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo

Best model found at epoch 5 with IoU **0.15269382297992706**
- Test -> {'loss': 7.514447212219238, 'accuracy': 0.3072086274623871, 'precision': 0.32009345293045044, 'recall': 0.3064907193183899, 'iou': 0.15269382297992706, 'dice': 0.25755012035369873} 
- Train -> {'loss': 0.3014364692180053, 'accuracy': 0.8693695503732433, 'precision': 0.8585607049257858, 'recall': 0.8547303559987441, 'iou': 0.7533220659131589, 'dice': 0.8546578676804252}

---

!python -m model.train --model=segnet --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo

Best model found at epoch 2 with IoU **0.1503361612558365**
- Test -> {'loss': 1.7664402723312378, 'accuracy': **0.4186750054359436**, 'precision': 0.4198734760284424, 'recall': 0.340816468000412, 'iou': 0.1503361612558365, 'dice': 0.21764075756072998} 
- Train -> {'loss': 0.35284809800593747, 'accuracy': 0.8999042689800263, 'precision': 0.7028773852016614, 'recall': 0.5207103727952294, 'iou': 0.44965030965597735, 'dice': 0.543692955893019} 


!python -m model.train --model=segnext --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo

Best model found at epoch 3 with IoU 0.1185356006026268
- Test -> {'loss': nan, 'accuracy': 0.26141107082366943, 'precision': 0.354095458984375, 'recall': 0.30558282136917114, 'iou': 0.1185356006026268, 'dice': 0.19854697585105896} 
- Train -> {'loss': 0.34756636094787846, 'accuracy': 0.9000919507897418, 'precision': 0.7070492643377055, 'recall': 0.5178124281375305, 'iou': 0.4484056125516477, 'dice': 0.5417394089957942}


!python -m model.train --model=effunet --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo

Best model found at epoch 7 with IoU 0.1293342113494873
- Test -> {'loss': 70.02923583984375, 'accuracy': 0.357814222574234, 'precision': 0.35728034377098083, 'recall': 0.33558809757232666, 'iou': 0.1293342113494873, 'dice': 0.19665953516960144} 
- Train -> {'loss': 0.23451341733984324, 'accuracy': 0.9167937628600908, 'precision': 0.7628422597180242, 'recall': 0.6032680595698564, 'iou': 0.5351591167242631, 'dice': 0.6515537053346634}

---

!python -m model.train --model=segnet --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 6 with IoU 0.1375158280134201
- Test -> {'loss': 2.6335384845733643, 'accuracy': 0.2886340022087097, 'precision': 0.35145312547683716, 'recall': 0.3953143060207367, 'iou': 0.1375158280134201, 'dice': 0.2248261272907257} 
- Train -> {'loss': 0.45793855332809946, 'accuracy': 0.7929329848807791, 'precision': 0.6493917973145195, 'recall': 0.5775502088277237, 'iou': 0.46413804979428, 'dice': 0.6008968547634457}


!python -m model.train --model=segnext --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 9 with IoU 0.1118076890707016
- Test -> {'loss': nan, 'accuracy': 0.3354230523109436, 'precision': 0.7784743309020996, 'recall': 0.3333333432674408, 'iou': 0.1118076890707016, 'dice': 0.16744910180568695} 
- Train -> {'loss': 0.1781236749952254, 'accuracy': 0.9250241735707159, 'precision': 0.8594355031200077, 'recall': 0.8342174358989881, 'iou': 0.7416711493678715, 'dice': 0.843979453522226}

!python -m model.train --model=effunet --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 8 with IoU **0.14282973110675812**
- Test -> {'loss': 9.176857948303223, 'accuracy': 0.2873983383178711, 'precision': 0.3887515664100647, 'recall': 0.41708478331565857, 'iou': 0.14282973110675812, 'dice': 0.23174062371253967} 
- Train -> {'loss': 0.1896417631727198, 'accuracy': 0.9210288980732794, 'precision': 0.8594678383806478, 'recall': 0.8222617488840352, 'iou': 0.7278115619783816, 'dice': 0.83291553310726}

---

!python -m model.train --model=segnet --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo

Best model found at epoch 9 with IoU 0.13208970427513123
- Test -> {'loss': 5.238126277923584, 'accuracy': 0.2277396023273468, 'precision': 0.2928285598754883, 'recall': 0.2704488933086395, 'iou': 0.13208970427513123, 'dice': 0.23164395987987518} 
- Train -> {'loss': 0.2280314856249353, 'accuracy': 0.9013227986252826, 'precision': 0.8724958323914072, 'recall': 0.8709581693877344, 'iou': 0.7824945299521736, 'dice': 0.8712221039378125}

!python -m model.train --model=segnext --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo

Best model found at epoch 9 with IoU 0.1118076890707016
- Test -> {'loss': nan, 'accuracy': 0.3354230523109436, 'precision': 0.7784743309020996, 'recall': 0.3333333432674408, 'iou': 0.1118076890707016, 'dice': 0.16744910180568695} 
- Train -> {'loss': 0.1742503414335458, 'accuracy': 0.924672460037729, 'precision': 0.9023418245108231, 'recall': 0.9015985310077668, 'iou': 0.8279219653295434, 'dice': 0.9017363856668058}

**!python -m model.train --model=effunet --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo**

Best model found at epoch 6 with IoU **0.19912287592887878**

- Test -> {'loss': 21.240909576416016, **'accuracy': 0.4339450001716614**, 'precision': 0.4373089671134949, 'recall': 0.38066643476486206, 'iou': 0.19912287592887878, 'dice': 0.3088137209415436} 
- Train -> {'loss': 0.1910024340385976, 'accuracy': 0.9174443864304086, 'precision': 0.893762079010839, 'recall': 0.8915848825288856, 'iou': 0.8129882931709289, 'dice': 0.8919412149035413}

---

!python -m model.train --model=segnet --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 2 with IoU 0.13847368955612183

- Test -> {'loss': 1.4231476783752441, 'accuracy': 0.40854567289352417, 'precision': 0.4963960647583008, 'recall': 0.3326534926891327, 'iou': 0.13847368955612183, 'dice': 0.19748926162719727} 
- Train -> {'loss': 0.812461589212003, 'accuracy': 0.6543437911116559, 'precision': 0.5136680683364039, 'recall': 0.3691250963055569, 'iou': 0.2521116081139316, 'dice': 0.3226535166087358}

!python -m model.train --model=segnext --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 2 with IoU **0.15319056808948517**

- Test -> {'loss': nan, 'accuracy': 0.3083769977092743, 'precision': 0.35885366797447205, 'recall': 0.35634180903434753, 'iou': 0.15319056808948517, 'dice': 0.2580549716949463} 
- Train -> {'loss': 0.799606141059295, 'accuracy': 0.6564411111499953, 'precision': 0.6571199818797734, 'recall': 0.3778862239226051, 'iou': 0.2599610084424848, 'dice': 0.3342016395667325}

!python -m model.train --model=effunet --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 6 with IoU 0.1409446746110916
Test -> {'loss': 5.354954719543457, 'accuracy': 0.28874486684799194, 'precision': 0.27732181549072266, 'recall': 0.29664498567581177, 'iou': 0.1409446746110916, 'dice': 0.23677051067352295} 
Train -> {'loss': 0.7765060178611589, 'accuracy': 0.664254694658777, 'precision': 0.5094418077365211, 'recall': 0.40856502535550493, 'iou': 0.2880035237773605, 'dice': 0.37630955045637876} 
