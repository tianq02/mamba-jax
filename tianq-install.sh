# spu jax setup
pip install spu flax jax
pip install tensorflow tensorflow-datasets keras scikit-learn 

# mamba-jax
pip install triton==2.1.0
pip install --no-deps absl-py 'jax-triton @ git+https://github.com/jax-ml/jax-triton.git@7778c47c0a27c0988c914dce640dec61e44bbe8c'
pip install torch==2.7.0 -f https://mirrors.aliyun.com/pytorch-wheels/cpu
# "spu flax jax" are put here to avoid version mismatch
pip install spu flax jax equinox einops huggingface_hub transformers matplotlib tqdm optax loguru wandb datasets