# AWS Public ECR for DJL inference base images
# 763104351884 is AWS's public ECR account ID
ARG DJL_BASE_REGISTRY=763104351884
ARG AWS_REGION=ap-northeast-2
FROM ${DJL_BASE_REGISTRY}.dkr.ecr.${AWS_REGION}.amazonaws.com/djl-inference:0.27.0-deepspeed0.12.6-cu121
COPY code/ /opt/ml/model/code/