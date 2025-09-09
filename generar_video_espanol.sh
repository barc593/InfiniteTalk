#!/bin/bash

# Script para generar video con audio en español usando InfiniteTalk
# Uso: ./generar_video_espanol.sh

echo "🎬 Generando video con InfiniteTalk en español..."

# Activar entorno conda
eval "$(conda shell.bash hook)"
conda activate multitalk

# Comando básico para generar video en español (resolución 480p)
python generate_infinitetalk.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --input_json examples/ejemplo_espanol.json \
    --size infinitetalk-480 \
    --sample_steps 40 \
    --mode streaming \
    --motion_frame 9 \
    --save_file video_espanol_480p

echo "✅ ¡Video generado! Busca el archivo video_espanol_480p.mp4"

# Si quieres generar en 720p (requiere más VRAM)
echo "🎬 ¿Quieres generar en 720p? Descomenta las siguientes líneas:"
echo "# python generate_infinitetalk.py \\"
echo "#     --ckpt_dir weights/Wan2.1-I2V-14B-480P \\"
echo "#     --wav2vec_dir 'weights/chinese-wav2vec2-base' \\"
echo "#     --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \\"
echo "#     --input_json examples/ejemplo_espanol.json \\"
echo "#     --size infinitetalk-720 \\"
echo "#     --sample_steps 40 \\"
echo "#     --mode streaming \\"
echo "#     --motion_frame 9 \\"
echo "#     --save_file video_espanol_720p"

# Si tu GPU tiene poca VRAM, usa este comando
echo "🔧 Para GPU con poca VRAM, usa:"
echo "python generate_infinitetalk.py \\"
echo "    --ckpt_dir weights/Wan2.1-I2V-14B-480P \\"
echo "    --wav2vec_dir 'weights/chinese-wav2vec2-base' \\"
echo "    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \\"
echo "    --input_json examples/ejemplo_espanol.json \\"
echo "    --size infinitetalk-480 \\"
echo "    --sample_steps 40 \\"
echo "    --num_persistent_param_in_dit 0 \\"
echo "    --mode streaming \\"
echo "    --motion_frame 9 \\"
echo "    --save_file video_espanol_lowvram"