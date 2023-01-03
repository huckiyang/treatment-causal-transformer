# treatment-causal-transformer

[Treatment Learning Causal Transformer for Noisy Image Classification](https://openaccess.thecvf.com/content/WACV2023/papers/Yang_Treatment_Learning_Causal_Transformer_for_Noisy_Image_Classification_WACV_2023_paper.pdf)


## Treatment Learning Transformer


```python
python causal_trans.py

```

Details in line (427) of the class Causal_Transformer

## Treatment Effect

Average Treatment Effect (ATE) batch evaluation

```python
eval.py
```

## Baseline setup:

    Task: Image Binary Classification (w/wo dogs)
    X: COCO Dataset image w/wo dogs (NICO benchmark) or w/wo noise (CPS)
    Z: Dog segments
    T: 1 if has car else 0 with some noise
    Y: 1 if has car else 0

<img src="https://github.com/huckiyang/treatment-causal-transformer/blob/main/tlt.png" width="900">

If you find this work is related to your research, please consider to cite the following work. Thank you!

```bib
@inproceedings{yang2023treatment,
  title={Treatment Learning Causal Transformer for Noisy Image Classification},
  author={Yang, Chao-Han Huck and Hung, I-Te and Liu, Yi-Chieh and Chen, Pin-Yu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6139--6150},
  year={2023}
}
```
