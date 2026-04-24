# Lightweight IoT Botnet Detection Research

This repository contains the ongoing work for a research project on **machine learning-based IoT botnet detection at the edge/gateway level**.

## Overview

The purpose of this research is to investigate whether malicious IoT network traffic can be detected effectively using **lightweight machine learning models** that are suitable for deployment on edge devices such as a gateway or Raspberry Pi.

IoT devices are common targets of botnet malware such as **Mirai** and **Gafgyt**. Many detection approaches rely on cloud-side analysis, but this may introduce extra latency and communication overhead. In this project, the focus is on exploring **gateway-level detection** as a faster and more practical alternative.

## Research Goals

The main goals of this project are:

- detect malicious IoT traffic using supervised machine learning
- build a baseline binary classification pipeline for benign vs malicious traffic
- compare lightweight ML models for detection performance
- evaluate models using standard classification metrics
- study whether edge/gateway-level deployment is feasible

## Current Direction

At the current stage, the research focuses on:

- using the **N-BaIoT** dataset as the first benchmark dataset
- starting with **binary classification**
- building and testing initial baseline models
- organizing experiments step by step before moving to larger-scale evaluation

## Planned Work

The planned work includes:

- dataset preparation and preprocessing
- training baseline models such as Decision Tree, Logistic Regression, and Random Forest
- comparing results using accuracy, precision, recall, F1-score, and confusion matrix
- extending experiments with more attack types or devices
- investigating lightweight models that are more suitable for edge deployment

## Notes

This repository is currently a **work in progress**.  
The experiments, code structure, and documentation will continue to evolve as the research progresses.
