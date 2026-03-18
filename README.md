# Identity-as-Presence
Official implementation of "Identity as Presence: Towards Appearance and Voice Personalized Joint Audio-Video Generation"

**Identity as Presence: Towards Appearance and Voice Personalized Joint Audio-Video Generation**<br>
[Yingjie Chen](https://chen-yingjie.github.io/), 
Shilun Lin,
Cai Xing,
Qixin Yan,
Wenjing Wang,
Dingming Liu,
Hao Liu,
Chen Li,
Jing LYU<br>

<p align="center">
<a href="https://arxiv.org/abs/TBD"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://chen-yingjie.github.io/projects/Identity-as-Presence/index.html"><img src="https://img.shields.io/badge/Project-Website-red"></a>
<a href=""><img src="https://img.shields.io/static/v1?label=Demo&message=Video&color=orange"></a>
<a href="" target='_blank'>
<img src="https://visitor-badge.laobi.icu/badge?page_id=TODO" />
</a>
</p>

<p align="center">
<img src="assets/teaser.png" width="88%" />
</p>

## 💡 Abstract
Recent advances have demonstrated compelling capabilities in synthesizing real individuals into generated videos, reflecting the growing demand for identity-aware content creation. Nevertheless, an openly accessible framework enabling fine-grained control over facial appearance and voice timbre across multiple identities remains unavailable. In this work, we present a unified and scalable framework for identity-aware joint audio-video generation, enabling high-fidelity and consistent personalization. Specifically, we introduce a data curation pipeline that automatically extracts identity-bearing information with paired annotations across audio and visual modalities, covering diverse scenarios from single-subject to multi-subject interactions. We further propose a flexible and scalable identity injection mechanism for single- and multi-subject scenarios, in which both facial appearance and vocal timbre act as identity-bearing control signals. Moreover, in light of modality disparity, we design a multi-stage training strategy to accelerate convergence and enforce cross-modal coherence. Experiments demonstrate the superiority of the proposed framework.

## 🔥 Updates
- (2026-03-18) The project page, demo video and technical report are released.

## 📑 TODO List
  - [x] Release inference code and model weights for single-subject scenarios
  - [] Release inference code and model weights for multi-subject scenarios

## Usage
### Environment
```shell
$ pip install -r requirements.txt
```

### Pretrained Weights

Please download the following pretrained models and place them in the `ckpts` directory:

1. **MMAudio**: https://huggingface.co/hkchengrex/MMAudio
2. **Wan2.2-TI2V-5B**: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
3. **Identity-as-Presence**: https://huggingface.co/echoanran/Identity-as-Presence

After downloading, ensure all model files are placed in the `ckpts` directory and properly configured.

### Inference
```shell
$ bash infer.sh
```
The results will be saved in `./results` directory.

## 🎥 Demo 

### Single-subject Personalized Generation
<table class="center">
<tr>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/1.mp3"></audio>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/2.mp3"></audio>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/3.mp3"></audio>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/4.mp3"></audio>
    </td>
</tr>
<tr>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/1.mp3"></audio>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/ba26a1dd-b8be-477c-b97d-cffb41c088ce" muted="false"></video>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/2.mp3"></audio>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/c26a9e54-8267-402c-b09a-c2229622a650" muted="false"></video>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/3.mp3"></audio>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/66f6113b-3fb2-4b29-951d-ba0bfbfa259c" muted="false"></video>
    </td>
    <td width=25% style="border: none">
      <audio controls src="assets/audios/single/4.mp3"></audio>
      <video controls autoplay loop src="https://github.com/user-attachments/assets/8ac02afd-01ae-414e-94f2-2559af025d14" muted="false"></video>
    </td>
</tr>
</table>

### Multi-subject Personalized Generation
<table class="center">
<tr>
    <td width="25%" style="border: none; vertical-align: top;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-1.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/1-1.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-2.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/1-2.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <video controls autoplay loop muted src="https://github.com/user-attachments/assets/f58bcb3b-5058-44a5-ad9f-bd4f8b50c89b" style="width: 100%; border-radius: 4px;"></video>
    </td>
    <td width="25%" style="border: none; vertical-align: top;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-1.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/2-1.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-2.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/2-2.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <video controls autoplay loop muted src="https://github.com/user-attachments/assets/29728cb6-aaeb-45be-a560-16d8238c3d9e" style="width: 100%; border-radius: 4px;"></video>
    </td>
    <td width="25%" style="border: none; vertical-align: top;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-1.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/3-1.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-2.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/3-2.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <video controls autoplay loop muted src="https://github.com/user-attachments/assets/7e24c06d-8929-427f-bd58-e38b0ebec16b" style="width: 100%; border-radius: 4px;"></video>
    </td>
    <td width="25%" style="border: none; vertical-align: top;">
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-1.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/4-1.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 4px; margin-bottom: 8px;">
        <img src="assets/images/2-2.png" style="width: 23%; object-fit: cover; border-radius: 4px;">
        <audio controls src="assets/audios/multi/4-2.mp3" style="width: 23%; height: 32px;"></audio>
      </div>
      <video controls autoplay loop muted src="https://github.com/user-attachments/assets/8191d7cd-89fa-4378-9b83-b9e8f964bf67" style="width: 100%; border-radius: 4px;"></video>
    </td>
</tr>
</table>


For more details, please refer to our [project page](https://chen-yingjie.github.io/projects/Identity-as-Presence/index.html).

## 🔗 Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{chen2026identity,
  title={Identity as Presence: Towards Appearance and Voice Personalized Joint Audio-Video Generation},
  author={Chen, Yingjie and Lin, Shilun and Xing, Cai and Yan, Qixin and Wang, Wenjing and Liu, Dingming and Liu, Hao and Li, Chen and LYU, Jing},
  journal={arXiv preprint arXiv:TBD},
  website={https://chen-yingjie.github.io/projects/Identity-as-Presence/index.html},
  year={2026}}
```

## Acknowledgements

We would like to thank the contributors to various open-source projects for their research and exploration.