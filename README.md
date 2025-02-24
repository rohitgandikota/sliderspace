# SliderSpace: Decomposing the Visual Capabilities of Diffusion Models
### [Project Website](https://sliderspace.baulab.info) | [Paper](https://arxiv.org/pdf/2502.01639) | [Trained Weights](https://sliderspace.baulab.info/sliderspace_weights/) | [Huggingface Demo](https://huggingface.co/spaces/baulab/SliderSpace)

Official implementation of "SliderSpace: Decomposing the Visual Capabilities of Diffusion Models"
***Unlock the creativity of diffusion models and get sliders to control the creative concepts!*** <br>

<div align='center'>
<img src = 'images/twitter_teaser.gif'>
</div>


## Setup
To set up your python environment:

```bash
conda create -n sliderspace python=3.10
conda activate sliderspace

git clone https://github.com/rohitgandikota/sliderspace.git
cd sliderspace
pip install -r requirements.txt
```

If you want to use Claude to expand your training prompts, you need to export your `ANTHROPIC_API_KEY` (ideally to your bashrc or windows environment variables)

## SDXL Sliderspace
To discover SliderSpace directions for a concept inside SDXL (and distilled versions) use the following script

```bash
python train_sdxl.py \
    --concept_prompts "picture of a spaceship" \
    --exp_name "spaceship" \
    --distilled_ckpt "dmd2" \
    --num_sliders 10 \
    --clip_total_samples 1000 \
```
You can train 32 sliders under *90 mins* on A6000 GPU! For comparison training 16 Concept Sliders can take upto 480 mins. For more stable discovery increase the `--clip_total_samples 10000` this will slow down the discovery process, but you will get far robust directions.

## FLUX Sliderspace
To discover SliderSpace directions for a concept inside FLUX models use the following script

```bash
python train_flux.py \
    --concept_prompts "picture of a wizard" \
    --exp_name "wizard" \
    --num_sliders 10 \
    --clip_total_samples 1000 \
    --clip_batch_size 1 \ 
```
FLUX discovery takes around *~120 mins* on A100 GPU to discover 32 directions. Increase `clip_batch_size` if you have enough VRAM. For more stable discovery increase the `--clip_total_samples 10000` this will slow down the discovery process, but you will get far robust directions.

## Evaluation and Inference with the Trained SliderSpace
Once you train your SliderSpace using about training scripts, you can run inference on the discovered sliders and discover the directions you trained:
- `notebooks/sdxl-inference.ipynb`: This is a simple and clean notebook for inference on SDXL sliderspace. You can make GIFs too! 
- `notebooks/flux-inference.ipynb`: This is a simple and clean notebook for inference on FLUX sliderspace. You can make GIFs too! 


## Citing our work
The preprint can be cited as follows
```bibtex
@inproceedings{gandikota2025sliderspace,
  title={SliderSpace: Decomposing the Visual Capabilities of Diffusion Models},
  author={Gandikota, Rohit and Wu, Zongze and Zhang, Richard and Bau, David and Shechtman, Eli and Kolkin, Nick},
  journal={arXiv preprint arXiv:2502.01639},
  year={2024}
}
```