# Pix2Next: Leveraging Vision Foundation Models for RGB to NIR Image Translation

> **Pix2Next** is a novel image-to-image (I2I) translation framework that generates **Near-Infrared (NIR)** images from RGB inputs. By integrating **Vision Foundation Models (VFM)** as a feature extractor and applying **cross-attention** within an **encoder-decoder** architecture, Pix2Next delivers **high-quality**, **high-fidelity** NIR image synthesis.

<p align="center">
  <img src="assets/detailed_architecture.png" width="1000" alt="Pix2Next Teaser">
</p>





---

## Paper

- **Title**: Pix2Next: Leveraging Vision Foundation Models for RGB to NIR Image Translation  
- **Authors**: Youngwan Jin, Incheol Park, Hanbin Song, Hyeongjin Ju, Yagiz Nalcakan, Shiho Kim   
- **Paper**: [https://arxiv.org/abs/2409.16706]
- **Project page**: [https://yonsei-stl.github.io/Pix2Next/]



---

## Quick start

### Installation 
First of all install torch with the appropriate CUDA version. 
```bash
cd pix2next/common/ops_dcn3/
python setup.py build install
```
### Datasets
Dataset download
[IDD-AW](https://iddaw.github.io/)
[Ranus](https://www.chrisgchoe.com/)

Data structure
```bash
Pix2Next
├── datasets
│   ├── ranus
│   │   ├── train_A (source: RGB)
│   │   ├── train_B (target: NIR)
│   │   ├── test_A  (source: RGB)
│   │   └── test_B  (target: NIR)
│   └── idd_aw
│       ...
│       
└── ...
```

### Training 
```bash
cd ~/pix2next/UNET/trainer/
python train.py
```
---

### Testing
```bash 
cd ~/pix2next/UNET/tester/
python test_unet.py
```
---

### Evaluation 
copy from test image folder, paste 새 evaluation folder 
```bash
cd ~/pix2next/UNET/evaluation/
python eval_all.py
```
---

## Performance 
### Ranus
<p align="center">
  <img src="assets/Ranus_qual.PNG" width="700" alt="Ranus_qual">
</p>

weight_file: [download](https://drive.google.com/file/d/1X9p3ht9-jhYB52R9_g3Ke8wyTgAGjDkg/view?usp=sharing/)

### IDDAW
<p align="center">
  <img src="assets/IDDAW_qual.PNG" width="700" alt="IDDAW_qual">
</p>

weight_file: [download](https://drive.google.com/file/d/1X9p3ht9-jhYB52R9_g3Ke8wyTgAGjDkg/view?usp=sharing/)

---

## visualization 

### Ranus
<p align="center">
  <img src="assets/Ranus_vis.PNG" width="1000" alt="Ranus">
</p>

### IDDAW
<p align="center">
  <img src="assets/IDDAW_vis.PNG" width="1000" alt="IDDAW">
</p>

### BDD100K zeroshot translation
<p align="center">
  <img src="assets/bdd100k_vis.PNG" width="1000" alt="bdd100k">
</p>

## References

[RANUS: RGB and NIR Urban Scene Dataset for Deep Scene Parsing](https://ieeexplore.ieee.org/document/8279453/)

[IDD-AW: A Benchmark for Safe and Robust Segmentation of Drive Scenes in Unstructured Traffic and Adverse Weather](https://iddaw.github.io/)

[InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://github.com/OpenGVLab/InternImage)

## Citation 

If this work is helpful for your research, please consider citing the following BibTeX entry.
```bibtex
@article{jin2024pix2next,
  title={Pix2Next: Leveraging Vision Foundation Models for RGB to NIR Image Translation},
  author={Jin, Youngwan and Park, Incheol and Song, Hanbin and Ju, Hyeongjin and Nalcakan, Yagiz and Kim, Shiho},
  journal={arXiv preprint arXiv:2409.16706},
  year={2024}
}
```
