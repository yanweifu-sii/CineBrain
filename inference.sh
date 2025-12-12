export CUDA_VISIBLE_DEVICES=0
python sample_brain_va.py --base configs/cogvideox_5b_lora_brain_va.yaml configs/infer_brain_va_5b_sub02.yaml --seed 42 \
 --jsonpath ./CineBrain/sub02_test.json
