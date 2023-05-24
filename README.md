# Instructions

1) Fork My Repository and Clone 
[https://github.com/lkurlandski/MalConv2](https://github.com/lkurlandski/MalConv2/tree/capstone)

2) Add the Following File
https://github.com/NeuromorphicComputationResearchProgram/MalConv2
- malconvGCT_nocat.checkpoint

3) Download and Install Anaconda
https://www.anaconda.com/products/distribution

3) Setup the Virtual Environment
```conda env create -f environment.yml```

4) Activate Environment Before Use
```conda activate MalConv2```

5) Install some other stuff
```pip install capstone==4.0.2 lief==0.12.2 pefile==2022.5.30```

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
