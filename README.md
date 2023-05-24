# Instructions

1) Fork My Repository and Clone 
https://github.com/lkurlandski/MalConv2

2) Add the Following Files
- malconvGCT_nocat.checkpoint
- main.py
- README.md

3) Download and Install Anaconda
https://www.anaconda.com/products/distribution

3) Setup the Virtual Environment
```conda env create -f environment.yml```

4) Install pytorch
```conda install pytorch==1.4.0 cpuonly -c pytorch```

5) Activate Environment Before Use
```conda activate MalConv2```

6) Run main

Query for help
```python main.py --help```

Evaluate the malicious files in ./data/malicious and store the results in ./results
```python main.py --output_path=./results --malware=./data/malicious```

Do the same as above but delete existing experiments from ./results first
```python main.py --output_path=./results --malware=./data/malicious --clean```

Include benign files as well
```python main.py --output_path=./results --malware=./data/malicious --goodware=./data/benign```

Perform the computations on GPU
```python main.py --output_path=./results --malware=./data/malicious --device="cuda:0"```

Use larger batch size for many files
```python main.py --output_path=./results --malware=./data/malicious --device="cuda:0" --batch_size=32```
