# MLOps-zoomcamp

## Table of contents
1. [General Info](#general-info)
2. [Introduction](#introduction)
3. [Experiment tracking](#experiment-tracking)
4. [Orchestration](#orchestration)

## General Info
This repository was created for MLOps Zoomcamp. I will upload my solution and briefly describe what i learned in every chapter. On this repository: [MLOps-Github](https://github.com/DataTalksClub/mlops-zoomcamp)
you can find all course resources. This course is completely free, so if you interested, feel free to join.

## Introduction
In the first week I had to setup enviroment and download required libraries. For example:
- conda,
- docker.

I use a virtual machine with Lubuntu. Then I got to know MLOps Maturity Model. We can divide this model into 5 parts:
- No Automation,
- DevOps but no MLOps,
- Automated Training,
- Automated Model Deployment,
- Full MLOps Automated Operations.

There is a [link](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model) to Microsoft description of this model.

After this steps I did a simple project with NY tripdata. The solution is in 01-Intro directory.

## Experiment tracking
In this chapter I got more knowledge about experiment tracking and MLFlow. I had to create a project with NY taxi dataset and did experiment tracking. I also learned about
data versioning, model registry. All code is in experiment tracking directory.

## Orchestration
That chapter was really challenging. I had a lot of problems with Mage.ai platform, but after all I run this website and now I am in the process of writing, so in the future I will share my code. 
The main tasks was:
* Run Mage
* Data preparation: ETL and feature engineering
* Training: sklearn models and XGBoost
* Observability: Monitoring and alerting
* Triggering: Inference and retraining
* Deploying: Running operations in production
