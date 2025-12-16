
<h1 align="center">My SOLUTION to <br/>
CS336: Language Modeling from Scratch <br/>
(Spring 2025 Version)</h1>

# Assignment 01: Tokenization & Language Modeling 
## Environment Setup & Data Download
We first need install the virtual environment manager `uv` to ensure reproducibility, portability, and ease of use.

```sh
pip install uv
# or
brew install uv
```

After installing `uv`, we can run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary. 

Or create and activate the environment manually using:
```sh
uv sync
source .venv/bin/activate
```

It will install all the dependencies specified in `pyproject.toml`.

Than we can download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

It will create a `data` folder in the current directory and download the required datasets into it.


## Part 1: BPE Tokenizer 


# Assignment 02: Flash Attention & Parallelism


# Assignment 03: Scaling Laws 

# Assignment 04: Data 

# Assignment 05: Alignment & RLHF (GRPO)
