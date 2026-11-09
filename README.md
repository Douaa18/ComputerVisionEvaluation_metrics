# Évaluation des modèles de vision par ordinateur

Classification · Détection d'objets · Segmentation sémantique · Segmentation d'instances

Tout le code est dans un seul notebook : [`evaluation_vision_avec_classification.ipynb`](evaluation_vision_avec_classification.ipynb).
Ce README documente le code section par section (les sections du notebook correspondent aux titres ci-dessous).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Douaa18/ComputerVisionEvaluation_metrics/blob/main/evaluation_vision_avec_classification.ipynb)

## Contenu du dépôt

```
.
├── README.md
├── requirements.txt
└── evaluation_vision_avec_classification.ipynb
```

## Exécution

**Google Colab (recommandé)** : cliquer sur le badge ci-dessus, activer le GPU (*Exécution > Modifier le type d'exécution > GPU*), puis *Exécution > Tout exécuter*.

**En local**

```bash
pip install -r requirements.txt
jupyter notebook evaluation_vision_avec_classification.ipynb
```

Au premier lancement, le notebook télécharge `annotations_trainval2017.zip` (~240 Mo) et 50 images de COCO val2017 dans `data/`. Sur CPU, comptez une dizaine de minutes pour 50 images. Paramètres modifiables dans la section 0 : `N_IMAGES` (50 par défaut, 200 pour des résultats plus stables) et `SEED` (0, pour la reproductibilité).

## Documentation du code

| Section | Tâche | Modèle / données | Métriques |
|---|---|---|---|
| 0 | Setup et données | COCO val2017 | - |
| 1 | Métriques de base | - | IoU, précision, rappel, F1, Dice |
| 2 | Classification d'images | Exemples chiffrés illustratifs | Accuracy, précision, rappel, F1, ROC-AUC, Top-k, Hamming loss |
| 3 | Détection d'objets | Faster R-CNN ResNet50-FPN | AP, mAP@0.5, mAP@[.5:.95], AP par taille, courbe PR, F1-confiance |
| 4 | Segmentation sémantique | DeepLabV3 ResNet50 | Pixel accuracy, IoU par classe, mIoU, Dice, matrice de confusion |
| 5 | Segmentation d'instances | Mask R-CNN ResNet50-FPN | Mask AP, comparaison box AP / mask AP |
| 6 | Synthèse | - | Tableau comparatif, analyse d'erreurs |

### Section 0 : Setup et données
Installe `pycocotools`, importe les bibliothèques, fixe `SEED`, `N_IMAGES`, `DEVICE` et `DATA_DIR`. Télécharge `instances_val2017.json` puis tire au hasard `N_IMAGES` images ayant au moins une annotation non « crowd ». Ces annotations (boîtes + masques) servent de vérité terrain pour les trois tâches COCO.

### Section 1 : Métriques de base
- `box_iou_np(a, b)` : matrice d'IoU entre deux ensembles de boîtes (format xyxy).
- `mask_iou_np(m1, m2)` : IoU entre deux masques binaires.
- `dice_np(m1, m2)` : coefficient de Dice entre deux masques.
- Illustration : une même boîte de référence face à des prédictions de plus en plus décalées, pour voir l'IoU diminuer.

### Section 2 : Classification d'images
Exemples **illustratifs** : aucun classifieur n'est entraîné, les matrices de confusion sont construites à la main pour montrer le calcul des métriques.
- **2.1 Binaire** (présence d'un zèbre) : TP/FN/FP/TN, accuracy, précision, rappel, spécificité, F1.
- **2.2 Multi-classe** (chat / chien / oiseau) : matrice N×N, métriques par classe, moyennes macro / micro / pondérée, Top-k accuracy.
- **2.3 Multi-label** (personne / voiture / chien / vélo) : exact match, Hamming loss, F1 micro et macro.

### Section 3 : Détection d'objets (Faster R-CNN)
- Inférence avec `fasterrcnn_resnet50_fpn` (seuil de score 0.05 pour obtenir la courbe PR complète), résultats au format COCO.
- `show_boxes` : visualisation vérité terrain (vert) / prédictions (rouge).
- **3.1** Évaluation officielle avec `pycocotools` (`coco_eval`) : AP@[.5:.95], AP@.5, AP@.75, AP small/medium/large.
- **3.2** Calcul manuel de l'AP (`match_class`, `average_precision`) pour comprendre le protocole COCO, et comparaison avec `pycocotools`.
- **3.3** Courbe précision / rappel / F1 selon le seuil de confiance, pour choisir un seuil en production.
- **3.4** AP selon la taille des objets.

### Section 4 : Segmentation sémantique (DeepLabV3)
- Inférence avec `deeplabv3_resnet50` (21 classes VOC, 0 = fond) ; correspondance classes COCO → VOC (`COCO2VOC`).
- Construction de la vérité terrain sémantique (`build_semantic_gt`), prédiction (`predict_semantic`), matrice de confusion cumulée (`confusion`), métriques (`seg_metrics`).
- Résultats : pixel accuracy, mIoU, mDice, fwIoU ; IoU par classe ; matrice de confusion normalisée.
- **Le piège de la pixel accuracy** : un faux modèle qui prédit toujours « fond » a une bonne pixel accuracy mais un mIoU quasi nul.

### Section 5 : Segmentation d'instances (Mask R-CNN)
- Inférence avec `maskrcnn_resnet50_fpn` ; masques encodés en RLE (format COCO).
- `show_instances` : visualisation des masques prédits.
- **5.1** Mask AP (`segm`) comparé au box AP (Faster R-CNN et Mask R-CNN).
- **5.2** Pourquoi le mask AP est plus bas : nuage de points IoU masque vs IoU boîte par instance.

### Section 6 : Synthèse
Tableau comparatif de toutes les tâches, puis analyse d'erreurs : les 3 images les moins bien segmentées (sémantique) et les détections très confiantes mais fausses.

## Remarques
- Les expériences réelles (modèles pré-entraînés `torchvision`, COCO val2017) concernent la détection et les deux segmentations. La classification est purement pédagogique.
- Avec seulement 50 images, les métriques sont indicatives et varient selon `SEED` ; augmenter `N_IMAGES` pour des valeurs plus stables.

## Références
- COCO dataset : https://cocodataset.org
- `pycocotools` : https://github.com/cocodataset/cocoapi
- Modèles `torchvision` : https://pytorch.org/vision/stable/models.html
