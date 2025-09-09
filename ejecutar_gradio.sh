#!/bin/bash

# Script para ejecutar InfiniteTalk con interfaz web Gradio
echo "🚀 Iniciando InfiniteTalk con interfaz web Gradio..."

# Activar entorno conda
eval "$(conda shell.bash hook)"
conda activate multitalk

echo "📱 La interfaz web estará disponible en:"
echo "   • Local: http://127.0.0.1:7860"
echo "   • Red: http://0.0.0.0:7860"
echo ""
echo "🎯 Funciones disponibles:"
echo "   • Subir tu propia imagen"
echo "   • Subir tu audio en español"
echo "   • Generar video con sincronización labial"
echo "   • Descargar resultado"
echo ""

# Ejecutar aplicación Gradio
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9

echo "✅ Aplicación web cerrada"