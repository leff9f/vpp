## установка драйверов ubuntu 16.04
1. скачать архив `https://drive.google.com/file/d/1PHSBb2tO_YYujuhSp8yLZqn8_DPyZIRe/view?usp=sharing`
2. удалить все прежнее версии:
    ```
    dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' '
    sudo apt-get purge $(dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' ')
    ```
3. `sudo ./cuda_10.1.168_418.67_linux.run` - снять галку с установки драйвера.
4. Установка драйвера:
    ```
    sudo apt-get update
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt install nvidia-430
    ```
5. Перезагрузка
6. Проверка
    ```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 430.64       Driver Version: 430.64       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:65:00.0  On |                  N/A |
    | 51%   56C    P8    16W / 250W |    710MiB / 11177MiB |      2%      Default |
    +-------------------------------+----------------------+----------------------+
    ```
7. Установка libcudnn:
    ```
    sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
    sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
    sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb
    ```

## установка драйверов на ubuntu 18.04
source: https://www.tensorflow.org/install/gpu#install_cuda_with_apt

1. удалить все прежнее версии:
    ```
    dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' '
    sudo apt-get purge $(dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' ')
    ```
2. Add NVIDIA package repositories
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-get update
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt-get update
    ```
3. Install NVIDIA driver
    ```
    sudo apt-get install --no-install-recommends nvidia-driver-450
    ```
4. Reboot. Check that GPUs are visible using the command: nvidia-smi
5. Install development and runtime libraries (~4GB)
    ```
    sudo apt-get install --no-install-recommends \
    cuda-11-0 \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0
    ```
    ```
    sudo apt-get install --no-install-recommends \
    cuda-10-1 \
    libcudnn7=7.6.5.32-1+cuda10.1  \
    libcudnn7-dev=7.6.5.32-1+cuda10.1
    ```
6. Install TensorRT. Requires that libcudnn7 is installed above.
    ```
    sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1
    ```


## установка драйверов на ubuntu 20.04
source: https://www.tensorflow.org/install/gpu#install_cuda_with_apt

1. удалить все прежнее версии:
    ```
    dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' '
    sudo apt-get purge $(dpkg -l | grep -E "cuda|nvidia|cudnn" | cut -d ' ' -f 3 | tr '\n' ' ')
    ```
2. Add NVIDIA package repositories
    ```
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
    sudo apt-get update
    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
    sudo apt-get update
    ```
3. Install NVIDIA driver
    ```
    sudo apt-get install --no-install-recommends nvidia-driver-460
    ```
4. Reboot. Check that GPUs are visible using the command: nvidia-smi
5. Install CUDA
    ```
    wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
    sudo sh cuda_11.2.0_460.27.04_linux.run
    ```
    Do what the instructions given in the summary say and add the given directories to your PATH and LD_LIBRARY_PATH. 
    For example by adding the following lines to your .bashrc, .zshrc, or whatever shell you are using:
    ```
    echo 'export PATH=/usr/local/cuda-11.2/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    ```
6. Reboot.Check installed CUDA: nvidia-smi
6. Install cuDNN
    Go here https://developer.nvidia.com/cudnn and follow the instructions
    ```
    tar -xzvf cudnn-11.0-linux-x64-v8.0.2.39.tgz
    ```
    ```
    sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
    sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```
7. Install tensorflow, must be > = 2.5.0
    ```
    pip install tensorflow
    pip install tensorflow-gpu
    ```
8. Check gpu 
   ```
   import tensorflow as tf
   gpu_available = tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )
   ```