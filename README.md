
## Beyond Illumination: Fine-Grained Detail Preservation in Extreme Dark Image Restoration [Paper](https://arxiv.org/abs/2508.03336) 

- *Tongshun Zhang, Pingping Liu, Zixuan Zhong, Zijian Zhang, Qiuzhan Zhou*
- *College of Computer Science and Technology, Jilin University*
- *Key Laboratory of Symbolic Computation and Knowledge Engineering of Ministry of Education*
- *College of Communication Engineering, Jilin University*

## 1. Abstract
Recovering fine-grained details in extremely dark images remains challenging due to severe structural information loss and noise corruption. Existing enhancement methods often fail to preserve intricate details and sharp edges, limiting their effectiveness in downstream applications like text and edge detection. To address these deficiencies, we propose an efficient dual-stage approach centered on detail recovery for dark images. In the first stage, we introduce a Residual Fourier-Guided Module (RFGM) that effectively restores global illumination in the frequency domain. RFGM captures inter-stage and inter-channel dependencies through residual connections, providing robust priors for high-fidelity frequency processing while mitigating error accumulation risks from unreliable priors. The second stage employs complementary Mamba modules specifically designed for textural structure refinement: (1) Patch Mamba operates on channel-concatenated non-downsampled patches, meticulously modeling pixel-level correlations to enhance fine-grained details without resolution loss. (2) Grad Mamba explicitly focuses on high-gradient regions, alleviating state decay in state space models and prioritizing reconstruction of sharp edges and boundaries. Extensive experiments on multiple benchmark datasets and downstream applications demonstrate that our method significantly improves detail recovery performance while maintaining efficiency. Crucially, the proposed modules are lightweight and can be seamlessly integrated into existing Fourier-based frameworks with minimal computational overhead.


### 4. Train
```
python train.py -opt ./options/train/.yml
```

### 5. Test
```
python test.py -opt ./options/test/.yml
```
