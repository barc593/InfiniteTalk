#!/bin/bash

# Setup automático de InfiniteTalk basado en recursos del sistema
# Detecta GPU/RAM y descarga modelos apropiados

echo "🤖 Setup automático de InfiniteTalk"
echo "=================================="

# Detectar recursos del sistema
echo "🔍 Analizando recursos del sistema..."

# RAM
RAM_TOTAL_GB=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
RAM_FREE_GB=$(free -m | awk 'NR==2{printf "%.0f", $7/1024}')
echo "💾 RAM: ${RAM_FREE_GB}GB libres de ${RAM_TOTAL_GB}GB totales"

# GPU Detection
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA no detectado. Este script es para GPUs NVIDIA."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

echo "🎮 GPU: $GPU_NAME"
echo "📊 VRAM: ${VRAM_FREE}MB libres de ${VRAM_TOTAL}MB totales"

# Determinar configuración óptima basada en recursos
echo ""
echo "💡 Determinando configuración óptima..."

# Configuración por defecto
QUANT_LEVEL=""
DOWNLOAD_FULL=false
VRAM_MODE=""
RESOLUTION="480"

# Lógica de decisión basada en VRAM
if [ "$VRAM_TOTAL" -ge 40000 ]; then
    # GPUs High-end (A100, H100, etc)
    echo "🚀 GPU High-end detectada (${VRAM_TOTAL}MB)"
    QUANT_LEVEL="none"
    DOWNLOAD_FULL=true
    VRAM_MODE="normal"
    RESOLUTION="720"
elif [ "$VRAM_TOTAL" -ge 24000 ]; then
    # RTX 4090, RTX 6000 Ada
    echo "🎮 GPU Gaming/Pro detectada (${VRAM_TOTAL}MB)"
    QUANT_LEVEL="fp8"
    DOWNLOAD_FULL=false
    VRAM_MODE="low"
    RESOLUTION="720"
elif [ "$VRAM_TOTAL" -ge 16000 ]; then
    # RTX 4080, RTX 3090
    echo "🎯 GPU Mid-range detectada (${VRAM_TOTAL}MB)"
    QUANT_LEVEL="fp8"
    DOWNLOAD_FULL=false
    VRAM_MODE="low"
    RESOLUTION="480"
elif [ "$VRAM_TOTAL" -ge 12000 ]; then
    # RTX 4070 Ti, RTX 3080
    echo "⚡ GPU Mid detectada (${VRAM_TOTAL}MB)"
    QUANT_LEVEL="int8"
    DOWNLOAD_FULL=false
    VRAM_MODE="low"
    RESOLUTION="480"
elif [ "$VRAM_TOTAL" -ge 8000 ]; then
    # RTX 4070, RTX 3070
    echo "🔧 GPU Entry detectada (${VRAM_TOTAL}MB)"
    QUANT_LEVEL="int8"
    DOWNLOAD_FULL=false
    VRAM_MODE="low"
    RESOLUTION="480"
else
    echo "❌ VRAM insuficiente (${VRAM_TOTAL}MB). Mínimo 8GB requerido."
    exit 1
fi

# Ajustar según RAM disponible
if [ "$RAM_TOTAL_GB" -lt 32 ] && [ "$QUANT_LEVEL" = "none" ]; then
    echo "⚠️  RAM limitada - Forzando cuantización"
    QUANT_LEVEL="fp8"
    DOWNLOAD_FULL=false
fi

# Mostrar configuración decidida
echo ""
echo "📋 Configuración automática:"
echo "   Cuantización: $QUANT_LEVEL"
echo "   Modelo completo: $([ "$DOWNLOAD_FULL" = true ] && echo "Sí" || echo "Solo cuantizado")"
echo "   Modo VRAM: $VRAM_MODE"
echo "   Resolución: ${RESOLUTION}p"

read -p "¿Continuar con esta configuración? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Setup cancelado"
    exit 0
fi

# Instalación de dependencias
echo ""
echo "📦 Instalando dependencias..."

# Dependencias básicas
pip install -r requirements.txt
pip install librosa "misaki[en]" "numpy<2.2,>=1.24"

# Flash Attention (crítico)
echo "⚡ Instalando Flash Attention..."
pip install flash-attn --no-build-isolation

# Crear directorio de pesos si no existe
mkdir -p weights

# Descarga de modelos base (siempre necesarios)
echo ""
echo "📥 Descargando modelos base..."

echo "   - Modelo Wan2.1 I2V..."
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P

echo "   - Encoder de audio..."
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base

# Descarga selectiva de modelos InfiniteTalk
echo "   - Modelos InfiniteTalk..."
if [ "$DOWNLOAD_FULL" = true ]; then
    echo "     Descargando modelo completo..."
    huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk
else
    # Solo descargar archivos específicos según cuantización
    echo "     Descargando solo modelos cuantizados..."
    
    # Archivos base necesarios
    huggingface-cli download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir ./weights/InfiniteTalk
    huggingface-cli download MeiGen-AI/InfiniteTalk README.md --local-dir ./weights/InfiniteTalk
    
    # Modelos cuantizados según configuración
    if [ "$QUANT_LEVEL" = "fp8" ]; then
        echo "     Descargando modelos FP8..."
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/infinitetalk_single_fp8.safetensors --local-dir ./weights/InfiniteTalk
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/infinitetalk_single_fp8.json --local-dir ./weights/InfiniteTalk
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/t5_fp8.safetensors --local-dir ./weights/InfiniteTalk
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/t5_map_fp8.json --local-dir ./weights/InfiniteTalk
    elif [ "$QUANT_LEVEL" = "int8" ]; then
        echo "     Descargando modelos INT8..."
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/infinitetalk_single_int8.safetensors --local-dir ./weights/InfiniteTalk
        huggingface-cli download MeiGen-AI/InfiniteTalk quant_models/infinitetalk_single_int8.json --local-dir ./weights/InfiniteTalk
    fi
fi

# Generar script de lanzamiento personalizado
echo ""
echo "📝 Generando script de lanzamiento..."

cat > launch_optimized.sh << EOF
#!/bin/bash
# Script generado automáticamente para tu configuración

echo "🚀 Lanzando InfiniteTalk optimizado para $GPU_NAME"

CMD_ARGS="--ckpt_dir weights/Wan2.1-I2V-14B-480P"
CMD_ARGS="\$CMD_ARGS --wav2vec_dir weights/chinese-wav2vec2-base"
CMD_ARGS="\$CMD_ARGS --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors"
CMD_ARGS="\$CMD_ARGS --motion_frame 9"

EOF

if [ "$QUANT_LEVEL" != "none" ]; then
cat >> launch_optimized.sh << EOF
CMD_ARGS="\$CMD_ARGS --quant $QUANT_LEVEL"
CMD_ARGS="\$CMD_ARGS --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_${QUANT_LEVEL}.safetensors"
EOF
fi

if [ "$VRAM_MODE" = "low" ]; then
cat >> launch_optimized.sh << EOF
CMD_ARGS="\$CMD_ARGS --num_persistent_param_in_dit 0"
EOF
fi

cat >> launch_optimized.sh << EOF

echo "Configuración: Cuantización=$QUANT_LEVEL, VRAM=$VRAM_MODE, Resolución=${RESOLUTION}p"
python app.py \$CMD_ARGS
EOF

chmod +x launch_optimized.sh

echo ""
echo "✅ Setup completado exitosamente!"
echo ""
echo "🎯 Tu configuración optimizada:"
echo "   GPU: $GPU_NAME (${VRAM_TOTAL}MB VRAM)"
echo "   RAM: ${RAM_TOTAL_GB}GB"
echo "   Cuantización: $QUANT_LEVEL"
echo "   Resolución: ${RESOLUTION}p"
echo ""
echo "🚀 Para ejecutar, usa:"
echo "   ./launch_optimized.sh"
echo ""
echo "📁 O manualmente:"
if [ "$QUANT_LEVEL" != "none" ]; then
    echo "   python app.py --quant $QUANT_LEVEL --quant_dir weights/InfiniteTalk/quant_models/infinitetalk_single_${QUANT_LEVEL}.safetensors --num_persistent_param_in_dit 0 --motion_frame 9 --ckpt_dir weights/Wan2.1-I2V-14B-480P --wav2vec_dir weights/chinese-wav2vec2-base --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors"
else
    echo "   python app.py --motion_frame 9 --ckpt_dir weights/Wan2.1-I2V-14B-480P --wav2vec_dir weights/chinese-wav2vec2-base --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors"
fi