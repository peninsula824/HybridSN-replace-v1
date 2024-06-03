#!/bin/bash

DATASET_PATH="./dataset"
CLASS_MAP_PATH="classification_maps"
MODEL_PATH="models"
REPORT_PATH="report"

declare -a NETWORKS=("PyResNet"
                    "SSRN"
                    "S3KAIResNet"
                    "ResNet"
                    "ContextualNet")

declare -a DATAFILES=("Indian_pines_corrected.mat"
                    "Indian_pines_gt.mat"
                    "PaviaU.mat"
                    "PaviaU_gt.mat"
                    "KSC.mat"
                    "KSC_gt.mat"
                    "Salinas_corrected.mat"
                    "Salinas_gt.mat")
declare -a DATAURL=("http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
                    "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
                    "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
                    "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"
                    "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat"
                    "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat"
                    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_corrected.mat"
                    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_gt.mat")

if [ ! -d "$DATASET_PATH" ]; then
  echo "Creating ${DATASET_PATH}..."
  mkdir "$DATASET_PATH"
fi
length=${#DATAFILES[@]}
for (( i = 0; i < length; i++ ));
do
    if [ -f "$DATASET_PATH/${DATAFILES[i]}" ]; then
        echo "${DATAFILES[i]} File exists..."
    else
        echo "${DATAFILES[i]} file doesn't exist..."
        wget "${DATAURL[i]}" -P "$DATASET_PATH"
    fi
done

for net in "${NETWORKS[@]}"
do
  if [ ! -d "$net/$CLASS_MAP_PATH" ]; then
    echo "Creating $net/$CLASS_MAP_PATH..."
    mkdir -p "$net/$CLASS_MAP_PATH"
  fi
  if [ ! -d "$net/$MODEL_PATH" ]; then
    echo "Creating $net/$MODEL_PATH..."
    mkdir -p "$net/$MODEL_PATH"
  fi
  if [ ! -d "$net/$REPORT_PATH" ]; then
    echo "Creating $net/$REPORT_PATH..."
    mkdir -p "$net/$REPORT_PATH"
  fi
done
