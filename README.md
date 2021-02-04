# imageScalingCUDA


#### Input

**1. image**  ( std::shared_ptr< cv::Mat > )

**2. AEye_result** ( SimpleBBoxPtr )

**3. bgr_planar_gpu_array** ( std::vector<gpu_shared_ptr<float>> )

   - output 변수

   - 형태 : [4 x outputHeight x outputWidth] vector , k개

        - k : (object_size / 4) + 1

**4. outputWidth** ( int )

**5. outputHeight** ( int )

--------------------------------------

#### Process

- [x] **gpuCrop**

    - bgr_packed_unchar_cpu 이미지를 gpuMemory에 복사

    - box offset 위치 정보대로 이미지 크롭

- [x] **gpuResize**

    - 2 methods
    
      - [x]  nearest neighbor interpolation

      - [x]  bilinear interpolation 



- [x] **packedToPlanar**

    - bgr_packed_gpu에서 bgr_planar_gpu 변환


- [x] **mergeImgs**

     - 배치 수만큼 bgr_planar_gpu 이미지 정합
