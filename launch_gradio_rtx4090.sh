#!/bin/bash

# Script para lanzar InfiniteTalk Gradio con detección automática de recursos
# Optimizado para RTX 4090

echo "🚀 Lanzando InfiniteTalk Gradio..."

# Detectar recursos del sistema
echo "🔍 Detectando recursos del sistema..."

# Detectar RAM
RAM_TOTAL_GB=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
RAM_FREE_GB=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')

echo "💾 RAM: ${RAM_FREE_GB}GB libres de ${RAM_TOTAL_GB}GB totales"

# Detectar GPU y VRAM
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi no encontrado. ¿Está instalado CUDA?"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

echo "🎮 GPU: $GPU_NAME"
echo "📊 VRAM: ${VRAM_FREE}MB libres, ${VRAM_USED}MB usados de ${VRAM_TOTAL}MB totales"

# Recomendaciones basadas en recursos
echo ""
echo "💡 Análisis de configuración:"

if [ "$RAM_TOTAL_GB" -lt 32 ]; then
    echo "⚠️  RAM limitada (${RAM_TOTAL_GB}GB) - Se requiere cuantización obligatoria"
    FORCE_QUANT=true
elif [ "$RAM_TOTAL_GB" -lt 64 ]; then
    echo "⚠️  RAM moderada (${RAM_TOTAL_GB}GB) - Cuantización recomendada"
    FORCE_QUANT=true
else
    echo "✅ RAM suficiente (${RAM_TOTAL_GB}GB) - Cuantización recomendada para estabilidad"
    FORCE_QUANT=true
fi

if [ "$VRAM_TOTAL" -lt 16000 ]; then
    echo "⚠️  VRAM limitada (${VRAM_TOTAL}MB) - Modo bajo VRAM obligatorio"
    LOW_VRAM=true
elif [ "$VRAM_TOTAL" -lt 24000 ]; then
    echo "⚠️  VRAM moderada (${VRAM_TOTAL}MB) - Modo bajo VRAM recomendado"
    LOW_VRAM=true
else
    echo "✅ VRAM suficiente (${VRAM_TOTAL}MB) - Modo bajo VRAM para compatibilidad"
    LOW_VRAM=true
fi

if [ "$VRAM_FREE" -lt 2000 ]; then
    echo "❌ Error: Solo ${VRAM_FREE}MB VRAM libre. Cierra otros procesos GPU primero."
    exit 1
fi

# Verificar dependencias críticas
echo ""
echo "📋 Verificando dependencias..."

python -c "import flash_attn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Flash Attention no instalado"
    echo "Ejecuta: pip install flash-attn --no-build-isolation"
    exit 1
else
    echo "✅ Flash Attention disponible"
fi

# Verificar modelos
if [ ! -f "weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors" ]; then
    echo "❌ Error: Modelo cuantizado fp8 no encontrado"
    echo "Ejecuta: huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk"
    exit 1
else
    echo "✅ Modelos cuantizados disponibles"
fi

# Construir comando basado en recursos detectados
CMD_ARGS=""
CMD_ARGS="$CMD_ARGS --ckpt_dir weights/Wan2.1-I2V-14B-480P"
CMD_ARGS="$CMD_ARGS --wav2vec_dir weights/chinese-wav2vec2-base"
CMD_ARGS="$CMD_ARGS --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors"
CMD_ARGS="$CMD_ARGS --motion_frame 9"

if [ "$FORCE_QUANT" = true ]; then
    CMD_ARGS="$CMD_ARGS --quant fp8"
    CMD_ARGS="$CMD_ARGS --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_fp8.safetensors"
    echo "🔧 Aplicando cuantización FP8"
fi

if [ "$LOW_VRAM" = true ]; then
    CMD_ARGS="$CMD_ARGS --num_persistent_param_in_dit 0"
    echo "🔧 Aplicando modo bajo VRAM"
fi

echo ""
echo "🎬 Configuración final:"
echo "   - GPU: $GPU_NAME"
echo "   - Cuantización: $([ "$FORCE_QUANT" = true ] && echo "FP8" || echo "Desactivada")"
echo "   - Modo VRAM: $([ "$LOW_VRAM" = true ] && echo "Bajo consumo" || echo "Normal")"
echo "   - Puerto: 8418"

echo ""
echo "🚀 Iniciando servidor Gradio..."

# Ejecutar comando construido dinámicamente
python app.py $CMD_ARGS

echo ""
echo "🎉 InfiniteTalk terminado!"