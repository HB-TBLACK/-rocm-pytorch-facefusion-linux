# -rocm-pytorch-facefusion-linux
Guía completa y probada para configurar FaceFusion con aceleración GPU AMD (ROCm) en Linux (Ubuntu 22.04), incluyendo la instalación de ROCm y PyTorch   optimizado para AMD.

# Guía Completa: Configuración de AMD ROCm, PyTorch y FaceFusion en Linux (Ubuntu 22.04)

Esta guía detalla los pasos para configurar un entorno de desarrollo completo para GPUs AMD en Linux, incluyendo el stack de ROCm, PyTorch con soporte ROCm, y la integración con FaceFusion.

**Sistema Operativo:** Ubuntu 22.04 LTS  
**GPU AMD:** Serie RDNA2/RDNA3 (ej. RX 7900 XTX)  
**Versiones Clave:** ROCm 6.4.1, Python 3.10, ONNX Runtime 1.21.0, PyTorch 2.0.1

---

## 0. Preparación del Sistema

Asegúrate de que tu sistema esté actualizado y listo para la instalación de controladores y bibliotecas.

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 1. Instalación de AMDGPU y ROCm

Instalaremos el controlador AMDGPU y el stack de ROCm completo, que incluye las herramientas necesarias para la computación GPGPU (HIP, MIOpen, etc.).

1. **Descarga el paquete `amdgpu-install`:**

   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/jammy/amdgpu-install_6.4.60401-1_all.deb
   ```

2. **Instala `amdgpu-install`:**

   ```bash
   sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
   ```

   *Durante la instalación, si se te pregunta sobre la sobrescritura de archivos de configuración (`amdgpu.list`, `rocm.list`), responde `y` para instalar la versión del desarrollador.*

3. **Actualiza la lista de paquetes de APT:**

   ```bash
   sudo apt update
   ```

4. **Instala el stack completo de ROCm:**

   ```bash
   sudo amdgpu-install --usecase=graphics,rocm,hip,mllib
   ```

   **¡REINICIA TU SISTEMA INMEDIATAMENTE DESPUÉS DE ESTE PASO!**  
   Es crucial para que los nuevos controladores y módulos del kernel se carguen correctamente.

---

## 2. Configuración del Entorno Conda (Python 3.10)

La compatibilidad de versiones es crítica. Crearemos un entorno Conda dedicado con Python 3.10, ya que `onnxruntime-rocm` 1.21.0 está compilado para esta versión.

1. **Desactiva cualquier entorno Conda activo:**

   ```bash
   conda deactivate
   ```

2. **Elimina cualquier entorno `facefusion` existente (si lo tienes):**

   ```bash
   conda env remove --name facefusion -y
   ```

3. **Crea un nuevo entorno `facefusion` con Python 3.10:**

   ```bash
   conda create --name facefusion python=3.10 -y
   ```

4. **Activa el nuevo entorno `facefusion`:**

   ```bash
   conda activate facefusion
   ```

   *Si `conda activate` no funciona o tu prompt no cambia, ejecuta `conda init`, **cierra y vuelve a abrir tu terminal**, y luego intenta `conda activate facefusion` de nuevo.*

---

## 3. Instalación de PyTorch con ROCm

Instalaremos PyTorch optimizado para ROCm dentro de tu entorno `facefusion`.

1. **Asegúrate de que tu entorno `(facefusion)` esté activado.**

2. **Instala PyTorch 2.0.1 para ROCm 6.4.1:**

   ```bash
   pip install torch==2.0.1+rocm6.4.1 torchvision==0.15.2+rocm6.4.1 torchaudio==2.0.2+rocm6.4.1 --index-url https://download.pytorch.org/whl/rocm6.0
   ```

   *Nota: Aunque la URL es `rocm6.0`, esta es la ruta oficial para las compilaciones de PyTorch compatibles con ROCm 6.x. La versión `2.0.1+rocm6.4.1` es la que se alinea con tu ROCm 6.4.1.*

---

## 4. Instalación de ONNX Runtime ROCm

Ahora instalaremos `onnxruntime-rocm` en el mismo entorno.

1. **Asegúrate de que tu entorno `(facefusion)` esté activado.**

2. **Instala `onnxruntime-rocm` 1.21.0:**

   ```bash
   pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/onnxruntime_rocm-1.21.0-cp310-cp310-manylinux_2_28_x86_64.whl
   ```

   *Si encuentras un error de "No queda espacio en el dispositivo", necesitas liberar espacio en tu disco. Considera limpiar `/tmp`, `~/.cache`, `~/.thumbnails`, y usar `conda clean --all` y `sudo apt clean`.*

3. **Resuelve el conflicto de `numpy`:**

   ```bash
   pip install numpy==1.26.4
   ```

4. **Instala las dependencias restantes de FaceFusion:**

   ```bash
   pip install gradio==5.25.2 gradio-rangeslider==0.0.8 onnx==1.17.0 opencv-python==4.11.0.86 psutil==7.0.0 tqdm==4.67.1 scipy==1.15.2
   ```

---

## 5. Configuración de Variables de Entorno (Persistente)

Para asegurar que PyTorch y ONNX Runtime detecten correctamente tu GPU AMD, añade estas variables a tu archivo `~/.bashrc` (o `~/.profile` si usas Zsh u otro shell).

1. **Abre tu archivo de configuración del shell:**

   ```bash
   nano ~/.bashrc
   ```

   (O `vim ~/.bashrc`, `gedit ~/.bashrc`, etc.)

2. **Añade las siguientes líneas al final del archivo:**

   ```bash
   # Configuración para AMD ROCm
   export HSA_OVERRIDE_GFX_VERSION=11.0.0
   export HCC_AMDGPU_TARGET=gfx1100
   export PYTORCH_ROCM_ARCH=gfx1100
   export TRITON_USE_ROCM=ON
   export ROCM_PATH=/opt/rocm-6.4.1
   export ROCR_VISIBLE_DEVICES=0
   export HIP_VISIBLE_DEVICES=0
   export USE_CUDA=0 # Asegura que no se intente usar CUDA
   export LD_LIBRARY_PATH=/opt/rocm-6.4.1/lib:/opt/rocm-6.4.1/hip/lib:$LD_LIBRARY_PATH
   ```

   *Nota: `gfx1100` es la arquitectura para la RX 7900 XTX. Si tienes otra GPU, verifica su arquitectura GFX.*

3. **Guarda el archivo y sal del editor.**

4. **Aplica los cambios al shell actual:**

   ```bash
   source ~/.bashrc
   ```

   *O cierra y vuelve a abrir tu terminal.*

---

## 6. Verificación de la Instalación

Verifica que ROCm y PyTorch estén funcionando correctamente con tu GPU.

1. **Verificar ROCm:**

   ```bash
   /opt/rocm-6.4.1/bin/roc-smi --version
   /opt/rocm-6.4.1/bin/rocminfo
   ```

   Deberías ver información sobre tu GPU y la versión de ROCm.

2. **Verificar PyTorch con ROCm:**

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

   Deberías ver `True` y el nombre de tu GPU.

3. **Verificar ONNX Runtime con ROCm:**

   ```bash
   python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```

   Deberías ver `ROCMExecutionProvider` en la salida.

---

## 7. Configuración y Ejecución de FaceFusion

Finalmente, configuraremos FaceFusion para que utilice el proveedor de ejecución de ROCm y descargaremos los modelos necesarios.

1. **Modifica `facefusion/facefusion/core.py`:**

   Este paso es crucial para asegurar que el modelo `face_swapper` se inicialice correctamente antes de las comprobaciones previas de FaceFusion.

   Abre el archivo `facefusion/facefusion/core.py` y **añade la siguiente línea** justo después de la definición de la función `def cli() -> None:`:

   ```python
   state_manager.set_item('face_swapper_model', 'inswapper_128_fp16')
   ```

   El bloque de código resultante debería verse así:

   ```python
   def cli() -> None:
       state_manager.set_item('face_swapper_model', 'inswapper_128_fp16') # <-- ¡Esta es la línea a añadir!
       if pre_check():
           signal.signal(signal.SIGINT, signal_exit)
           program = create_program()
       # ... el resto del código ...
   ```

2. **Configura `facefusion.ini`:**

   Modifica el archivo `facefusion/facefusion.ini` para establecer el proveedor de ejecución y los modelos predeterminados.

   ```
   [execution]
   execution_device_id =
   execution_providers = rocm
   execution_thread_count =
   execution_queue_count =

   [face_detector]
   face_detector_model = retinaface
   face_detector_size =
   face_detector_angles =
   face_detector_score =

   [face_landmarker]
   face_landmarker_model = peppa_wutz
   face_landmarker_score =

   [processors]
   face_swapper_model = inswapper_128_fp16

   [misc]
   log_level = debug
   ```

   *Puedes editar el archivo manualmente o usar comandos `replace` si lo prefieres.*

3. **Descarga los modelos de FaceFusion:**

   ```bash
   python facefusion/facefusion.py force-download
   ```

4. **Ejecuta FaceFusion con ROCm:**

   **Reemplaza `/ruta/a/tu/imagen_origen.jpg` y `/ruta/a/tu/video_destino.mp4` con las rutas reales de tus archivos.**

   ```bash
   python facefusion/facefusion.py run --execution-providers rocm -s /ruta/a/tu/imagen_origen.jpg -t /ruta/a/tu/video_destino.mp4
   ```

   Si todo está configurado correctamente, deberías ver un mensaje como `Running on local URL: http://127.0.0.1:7860`, indicando que la interfaz web se ha iniciado y está utilizando tu GPU AMD.

---
