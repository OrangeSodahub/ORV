# ORV: 4D Occupancy-centric Robot Video Generation

[[`Project Page`](https://orangesodahub.github.io/ORV/)] [[`arXiv`](http://arxiv.org/abs/2406.XXXXX)] [[`pdf`](https://arxiv.org/pdf/2506.XXXXX)] [[`BibTeX`](#BibTex)] [[`License`](https://github.com/OrangeSodahub/ORV?tab=MIT-1-ov-file)]


## About
TL;DR We propose ORV, a novel framework of robot video generation with the geometry guidance of 4D occupancy, which achieves higher control precision, shows strong generalizations, performs multi-view consistent videos generation and conducts simulation-to-real data transfer.

<img src="assets/teaser.png" width="100%"/>
<details>
<summary><strong>Abstract</strong></summary>
Acquiring real-world robotic simulation data through teleoperation is notoriously time-consuming and labor-intensive. Recently, action-driven generative models have gained widespread adoption in robot learning and simulation, as they eliminate safety concerns and reduce maintenance efforts. However, the action sequences used in these methods often result in limited control precision and poor generalization due to their globally coarse alignment. To address these limitations, we propose ORV, an Occupancy-centric Robot Video generation framework, which utilizes 4D semantic occupancy sequences as a fine-grained representation to provide more accurate semantic and geometric guidance for video generation. By leveraging occupancy-based representations, ORV enables seamless translation of simulation data into photorealistic robot videos, while ensuring high temporal consistency and precise controllability. Furthermore, our framework supports the simultaneous generation of multi-view videos of robot gripping operations—an important capability for downstream robotic learning tasks. Extensive experimental results demonstrate that ORV consistently outperforms existing baseline methods across various datasets and sub-tasks.
</details>
<!-- 
## BibTeX
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{yang2025orv,
    title={ORV: 4D Occupancy-centric Robot Video Generation},
    author={Yang, Xiuyu and Li, Bohan and Xu, Shaocong and Wang, Nan and Ye, Chongjie and Chen Zhaoxi and Qin, Minghan and Ding Yikang and Jin, Xin and Zhao, Hang and Zhao, Hao},
    journal={arXiv preprint arXiv:2506.XXXXX},
    year={2025}
}
``` -->

## TODO

- [ ] Release arXiv technique report
- [ ] Release full codes
- [ ] Release instructions for data processing
- [ ] Release processed data


## Acknowledgement

Thansk for these excellent opensource works and models: [CogVideoX](https://github.com/THUDM/CogVideo); [DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader); [diffusers](https://github.com/huggingface/diffusers);.
