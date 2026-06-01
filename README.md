
## [AAAI 2026] Beyond Illumination: Fine-Grained Detail Preservation in Extreme Dark Image Restoration [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/38276/42238) 

- *Tongshun Zhang, Pingping Liu, Zixuan Zhong, Zijian Zhang, Qiuzhan Zhou*
- *College of Computer Science and Technology, Jilin University*
- *Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education*
- *College of Communication Engineering, Jilin University*

## 1. Abstract
Recovering fine-grained details in extremely dark images remains challenging due to severe structural information loss and noise corruption. Existing enhancement methods often fail to preserve intricate details and sharp edges, limiting their effectiveness in downstream applications like text and edge detection. To address these deficiencies, we propose an efficient dual-stage approach centered on detail recovery for dark images. In the first stage, we introduce a Residual Fourier-Guided Module (RFGM) that effectively restores global illumination in the frequency domain. RFGM captures inter-stage and inter-channel dependencies through residual connections, providing robust priors for high-fidelity frequency processing while mitigating error accumulation risks from unreliable priors. The second stage employs complementary Mamba modules specifically designed for textural structure refinement: (1) Patch Mamba operates on channel-concatenated non-downsampled patches, meticulously modeling pixel-level correlations to enhance fine-grained details without resolution loss. (2) Grad Mamba explicitly focuses on high-gradient regions, alleviating state decay in state space models and prioritizing reconstruction of sharp edges and boundaries. Extensive experiments on multiple benchmark datasets and downstream applications demonstrate that our method significantly improves detail recovery performance while maintaining efficiency. Crucially, the proposed modules are lightweight and can be seamlessly integrated into existing Fourier-based frameworks with minimal computational overhead.

## 2. Over-all-Architecture
![Over-all-Architecture](<img width="1457" height="525" alt="image" src="https://github.com/user-attachments/assets/f5ccc863-6ed1-4c09-b9d4-41ebf41af52a" />
)


## 3. Datasets
- LOL-real and LOL-sys can be found in [here](https://github.com/flyywh/SGM-Low-Light).
- LSRW-Huawei and LSRW-Nikon can be found in [here](https://github.com/JianghaiSCU/R2RNet).

### 4. Train
```
python train.py -opt ./options/train/.yml
```

### 5. Test
```
python test.py -opt ./options/test/.yml
```

## Citation Information
If you find the project useful, please cite:  

```bibtex  
@inproceedings{zhang2026beyond,
  title={Beyond Illumination: Fine-Grained Detail Preservation in Extreme Dark Image Restoration},
  author={Zhang, Tongshun and Liu, Pingping and Zhong, Zixuan and Zhang, Zijian and Zhou, Qiuzhan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={15},
  pages={12789--12797},
  year={2026}
}

