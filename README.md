# GUARD: Dual-Agent Backdoor Defense for Chain-of-Thought in Code Generation

This repository contains the code and datasets for GUARD:

**GUARD: Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**  
Naizhu Jin†, Zhong Li†, Tian Zhang†, Qingkai Zeng†  
†State Key Laboratory for Novel Software Technology, Nanjing University  

## Overview

GUARD is a dual-agent defense framework designed to detect and repair backdoor attacks in Chain-of-Thought (CoT) reasoning for code generation models. It includes:

- **GUARD-Judge**: Detects suspicious CoT steps via correctness checks and pattern analysis.  
- **GUARD-Repair**: Rewrites malicious CoTs with secure alternatives using retrieval-augmented generation.

## Features

- Defends against stealthy CoT backdoor attacks like SABER.
- Compatible with popular LLMs (e.g., DeepSeek, Qwen).
- Maintains code generation performance while improving security.

## Evaluation

We evaluate GUARD on HumanEval-CoT and OpenEval-CoT using:
- CoT quality: BLEU, METEOR, ROUGE-L  
- Code quality: Pass@1  
- Security: Attack Success Rate (ASR)

*Please modify API KEYs paths in the code files before running.*
