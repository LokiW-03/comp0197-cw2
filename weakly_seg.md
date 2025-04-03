# Weakly segmentation performance

!python -m model.train --model=segnet --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo

Best model found at epoch 4 with IoU 0.3187313675880432
- Test -> {'loss': 1.4144318103790283, 'accuracy': 0.582002580165863, 'precision': 0.4415043294429779, 'recall': 0.43940675258636475, 'iou': 0.3187313675880432, 'dice': 0.43287891149520874} 
- Train -> {'loss': 0.468969074021215, 'accuracy': 0.7971132211063219, 'precision': 0.7796539135601209, 'recall': 0.7720229721587637, 'iou': 0.6441861023073611, 'dice': 0.7739881131959998} 

!python -m model.train --model=segnext --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo



!python -m model.train --model=effunet --pseudo_path=cam/saved_models/resnet50_pet_cam_pseudo.pt --pseudo

Best model found at epoch 1 with IoU 0.23939715325832367
Test -> {'loss': 4.543310165405273, 'accuracy': 0.5866038799285889, 'precision': 0.3479088246822357, 'recall': 0.34962138533592224, 'iou': 0.23939715325832367, 'dice': 0.32456374168395996} 
Train -> {'loss': 0.5187347393968831, 'accuracy': 0.7740219732989435, 'precision': 0.7529080398704695, 'recall': 0.7415468709624332, 'iou': 0.6074814882615338, 'dice': 0.7389325870120007} 

---

!python -m model.train --model=segnet --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo

Best model found at epoch 4 with IoU 0.20799994468688965
- Test -> {'loss': 1.0903692245483398, 'accuracy': 0.6209691762924194, 'precision': 0.22370511293411255, 'recall': 0.3179279565811157, 'iou': 0.20799994468688965, 'dice': 0.25734633207321167} 
- Train -> {'loss': 0.35301534663075984, 'accuracy': 0.9000262252662493, 'precision': 0.7023315774357837, 'recall': 0.5264768785756567, 'iou': 0.45331913777019667, 'dice': 0.5477068022541378} 



!python -m model.train --model=segnext --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo




!python -m model.train --model=effunet --pseudo_path=cam/saved_models/efficientnet_pet_scorecam_pseudo.pt --pseudo



---

!python -m model.train --model=segnet --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo



!python -m model.train --model=segnext --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo



!python -m model.train --model=effunet --pseudo_path=crm_models/resnet_drs_pet_scorecam_crm_pseudo.pt --pseudo



---

!python -m model.train --model=segnet --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo

Best model found at epoch 6 with IoU 0.3062465786933899
- Test -> {'loss': 1.9105478525161743, 'accuracy': 0.5524827241897583, 'precision': 0.4345264434814453, 'recall': 0.44297951459884644, 'iou': 0.3062465786933899, 'dice': 0.42748206853866577} 
- Train -> {'loss': 0.2641829153765803, 'accuracy': 0.8877702951431274, 'precision': 0.8552818404591602, 'recall': 0.8526355370231297, 'iou': 0.7574485462644825, 'dice': 0.8533654218134673} 


!python -m model.train --model=segnext --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo

Best model found at epoch 1 with IoU 0.1599893420934677
- Test -> {'loss': nan, 'accuracy': 0.3310995101928711, 'precision': 0.4120316505432129, 'recall': 0.38188108801841736, 'iou': 0.1599893420934677, 'dice': 0.2680383324623108} 
- Train -> {'loss': 0.6119007831034453, 'accuracy': 0.7341527769099111, 'precision': 0.6737939176352128, 'recall': 0.6356321145658907, 'iou': 0.5004591315984726, 'dice': 0.6205822965373163}

!python -m model.train --model=effunet --pseudo_path=crm_models/resnet_drs_pet_gradcampp_crm_pseudo.pt --pseudo

Best model found at epoch 8 with IoU 0.30324143171310425
- Test -> {'loss': 2.611558437347412, 'accuracy': 0.5893977880477905, 'precision': 0.44228243827819824, 'recall': 0.43923911452293396, 'iou': 0.30324143171310425, 'dice': 0.4194931983947754} 
- Train -> {'loss': 0.16331261364014252, 'accuracy': 0.9296653317368548, 'precision': 0.9091626221718996, 'recall': 0.9080381362334542, 'iou': 0.8377998095491658, 'dice': 0.9080680445484494} 

---

!python -m model.train --model=segnet --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo


!python -m model.train --model=segnext --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo



!python -m model.train --model=effunet --pseudo_path=crm_models/efficientnet_pet_scorecam_crm_pseudo.pt --pseudo

Best model found at epoch 1 with IoU 0.13522249460220337
- Test -> {'loss': 12.586511611938477, 'accuracy': 0.23860634863376617, 'precision': 0.3392791748046875, 'recall': 0.3502049148082733, 'iou': 0.13522249460220337, 'dice': 0.23040670156478882} 
- Train -> {'loss': 0.4022582315880319, 'accuracy': 0.8298588893983675, 'precision': 0.7064186532860217, 'recall': 0.6163677739060444, 'iou': 0.5041460804317308, 'dice': 0.626671334841977}


