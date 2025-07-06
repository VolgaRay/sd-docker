

set HF token to access (dont forget to accept licence for model https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo )
`export HF_TOKEN=<token>`

login in docker
`huggingface-cli login`

HF cache will be strored in user dir and mounted to docker
`mkdir ~/.cache/huggingface`
