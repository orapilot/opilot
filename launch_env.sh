#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="10.1"
fi

export STAGING_ROOT="/data/safe_staging"

export USE_WEBCAM=1
#export USE_THNEED=1
#export THNEED_DEBUG=1
#export DEBUG_CAM=1
#export ROADCAM_ID=$(ls -l /dev/v4l/by-id/ |grep index0|grep Generic|awk -F'video' '{print $NF}')
export ROADCAM_ID=$(ls -l /dev/v4l/by-id/ |grep index0|grep Sonix|awk -F'video' '{print $NF}')
#export WIDECAM_ID=$(ls -l /dev/v4l/by-id/ |grep index0|grep Sonix|awk -F'video' '{print $NF}')
#export DRIVERCAM_ID=$(ls -l /dev/v4l/by-id/ |grep index0|grep Sonix|awk -F'video' '{print $NF}')
#export DRIVERCAM_ID=2
#export ROADCAM_ID=1
export FINGERPRINT="HONDA CIVIC 2016"
#export FINGERPRINT="TOYOTA COROLLA HYBRID TSS2 2019"

#export DISPLAY=:0.0
export SIMULATION=1
export NOSENSOR=1
export GPU=1
export SKIP_FW_QUERY=1

#export NAV=1
#export MAPBOX_TOKEN="sk.eyJ1IjoiYmlyZHpoYW5nIiwiYSI6ImNsbjBna2piNDAxOWEycW9pd3F6bXdqdWIifQ.lXSixDAlFlidB87eg5ashg"

export GST_VIDEO_CONVERT_USE_RGA=1
export GST_MPP_VIDEODEC_DEFAULT_ARM_AFBC=1

