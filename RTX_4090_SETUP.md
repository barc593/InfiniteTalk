# InfiniteTalk Setup Guide for RTX 4090

Esta guía documenta la configuración exitosa de InfiniteTalk en una GPU RTX 4090 con optimizaciones específicas para evitar errores de memoria.

## Hardware Probado
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)  
- **RAM**: 64GB DDR4
- **OS**: Linux Ubuntu

## Problemas Encontrados y Soluciones

### 1. Error de Memoria RAM (OOM Killer)
**Problema**: El modelo de 14B parámetros consume >55GB RAM durante la carga inicial, causando que el sistema mate el proceso.

**Error típico**:
```bash
Killed  # Exit code 137
```

**Solución**: Usar cuantización FP8 que reduce significativamente el uso de memoria.

### 2. Error CUDA Out of Memory
**Problema**: Durante la generación, múltiples procesos compiten por VRAM.

**Error típico**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 644.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 336.31 MiB is free.
```

**Solución**: Combinar cuantización FP8 con modo bajo VRAM.

## Comando que Funciona - RTX 4090

### ✅ Gradio Interface (RECOMENDADO)
```bash
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9
```

### ✅ Script Directo  
```bash
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir weights/chinese-wav2vec2-base \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/ejemplo_espanol.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --quant fp8 \
    --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors \
    --motion_frame 9 \
    --num_persistent_param_in_dit 0 \
    --save_file infinitetalk_res_4090
```

## Parámetros Clave para RTX 4090

| Parámetro | Valor | Propósito |
|-----------|-------|-----------|
| `--quant` | `fp8` | Reduce uso de memoria RAM/VRAM |
| `--quant_dir` | `weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors` | Modelo cuantizado |
| `--num_persistent_param_in_dit` | `0` | Modo bajo VRAM |
| `--size` | `infinitetalk-480` | Resolución optimizada |
| `--motion_frame` | `9` | Frames de movimiento |

## Instalación de Dependencias Críticas

### Flash Attention (Requerido)
```bash
pip install flash-attn --no-build-isolation
```

### Dependencias del Sistema
```bash
pip install -r requirements.txt
pip install librosa misaki[en]

# Ajustar NumPy para compatibilidad
pip install "numpy<2.2,>=1.24"
```

## Descarga de Modelos (25GB+ total)
```bash
# Modelo base (14B parámetros)
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P

# Encoder de audio
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# Pesos de InfiniteTalk (incluye modelos cuantizados)
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
```

## Resultados de Rendimiento

### Tiempos de Carga
- **Sin cuantización**: Falla por OOM
- **Con FP8**: ~3 minutos para cargar todos los modelos

### Tiempos de Generación
- **Pasos de inferencia**: ~70 segundos por paso
- **Video de ~5 segundos**: ~8 minutos (8 pasos)
- **Memoria GPU utilizada**: ~12-15GB VRAM

### Uso de Recursos
- **RAM**: ~30GB durante generación (vs >55GB sin cuantizar)
- **VRAM**: ~12GB durante generación
- **CPU**: Utilización moderada

## Configuración de Archivos

### JSON de Entrada (ejemplo_espanol.json)
```json
{
    "prompt": "Una mujer está hablando apasionadamente al micrófono en un estudio de grabación...",
    "cond_video": "examples/single/ref_image.png",
    "cond_audio": {
        "person1": "examples/single/1.wav"
    }
}
```

## Solución de Problemas

### Si falla con "Killed"
- Asegúrate de usar `--quant fp8`
- Cierra otros procesos que usen mucha RAM

### Si falla con CUDA OOM
- Usa `--num_persistent_param_in_dit 0`
- Verifica que no haya otros procesos usando la GPU

### Si la interfaz no carga
- Revisa que el puerto 8418 esté libre
- Comprueba los logs para errores de dependencias

## Estado de Funcionalidad

✅ **Funciona**: Gradio con FP8 + Low VRAM  
✅ **Funciona**: Script directo con FP8 + Low VRAM  
❌ **No funciona**: Sin cuantización (OOM)  
❌ **No funciona**: Solo Low VRAM sin FP8 (CUDA OOM)

## Notas Importantes

1. **Flash Attention es obligatorio** - sin él, el rendimiento es muy pobre
2. **La cuantización FP8 es esencial** - sin ella, falla por memoria
3. **Los tiempos son largos** - cada video toma varios minutos
4. **Calidad**: FP8 mantiene buena calidad con menor uso de memoria
5. **Compatibilidad**: Funciona con múltiples procesos simultáneos

## Comandos de Verificación

```bash
# Verificar GPU
nvidia-smi

# Verificar memoria disponible  
free -h

# Verificar que Flash Attention esté instalado
python -c "import flash_attn; print('Flash Attention disponible')"

# Verificar archivos descargados
ls -la weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors
```

---

**Fecha de prueba**: 2025-09-09  
**Versión de InfiniteTalk**: Commit más reciente  
**Status**: ✅ Funcionando correctamente