# ffa_adversarial


## Details

### Base accuracy
|                   | Forward-Forward   | Linear    | Conv  |
|-------------------|-------------------|-----------|-------|
| Forward-Forward   | 0.904             |           |       |
| Linear            |                   | 0.887     |       |
| Conv              |                   |           | 0.927 |

### MNIST Saliency Map
#### FFA
|   Digit   |   FFA             |    LinearNet     |   ConvNet |
|-----------|-------------------| -----------------|-----------|
|   0       | <img src = "./code/saliency_maps/ffa/0.png" width=140> | <img src = "./code/saliency_maps/linear/0.png" width=140> | <img src = "./code/saliency_maps/conv/0.png" width=140> | 
|   1       | <img src = "./code/saliency_maps/ffa/1.png" width=140> | <img src = "./code/saliency_maps/linear/1.png" width=140> | <img src = "./code/saliency_maps/conv/1.png" width=140> |
|   2       | <img src = "./code/saliency_maps/ffa/2.png" width=140> | <img src = "./code/saliency_maps/linear/2.png" width=140> | <img src = "./code/saliency_maps/conv/2.png" width=140> |
|   3       | <img src = "./code/saliency_maps/ffa/3.png" width=140> | <img src = "./code/saliency_maps/linear/3.png" width=140> | <img src = "./code/saliency_maps/conv/3.png" width=140> |
|   4       | <img src = "./code/saliency_maps/ffa/4.png" width=140> | <img src = "./code/saliency_maps/linear/4.png" width=140> | <img src = "./code/saliency_maps/conv/4.png" width=140> |
|   5       | <img src = "./code/saliency_maps/ffa/5.png" width=140> | <img src = "./code/saliency_maps/linear/5.png" width=140> | <img src = "./code/saliency_maps/conv/5.png" width=140> |
|   6       | <img src = "./code/saliency_maps/ffa/6.png" width=140> | <img src = "./code/saliency_maps/linear/6.png" width=140> | <img src = "./code/saliency_maps/conv/6.png" width=140> |
|   7       | <img src = "./code/saliency_maps/ffa/7.png" width=140> | <img src = "./code/saliency_maps/linear/7.png" width=140> | <img src = "./code/saliency_maps/conv/7.png" width=140> |
|   8       | <img src = "./code/saliency_maps/ffa/8.png" width=140> | <img src = "./code/saliency_maps/linear/8.png" width=140> | <img src = "./code/saliency_maps/conv/8.png" width=140> |
|   9       | <img src = "./code/saliency_maps/ffa/9.png" width=140> | <img src = "./code/saliency_maps/linear/9.png" width=140> | <img src = "./code/saliency_maps/conv/9.png" width=140> |

### FGSM perturbations
|                   | Forward-Forward   | Linear    | Conv  |
|-------------------|-------------------|-----------|-------|
| Forward-Forward   | 0.634             | 0.856     | 0.885 |
| Linear            | 0.812             | 0.017     | 0.879 |
| Conv              | 0.857             | 0.891     | 0.364 |

### PGD perturbations
|                   | Forward-Forward   | Linear    | Conv  |
|-------------------|-------------------|-----------|-------|
| Forward-Forward   | 0.452             | 0.864     | 0.894 |
| Linear            | 0.809             | 0.008     | 0.867 |
| Conv              | 0.876             | 0.891     | 0.052 |