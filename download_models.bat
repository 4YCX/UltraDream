@echo off
chcp 65001 >nul
echo 正在创建 models 文件夹...
mkdir models 2>nul

echo 正在下载 decoder 模型...
curl -L -o models\new_decoder.pth https://hf-mirror.com/laion/DALLE2-PyTorch/resolve/main/decoder/v1.0.2/latest.pth

echo 正在下载 second decoder 模型...
curl -L -o models\second_decoder.pth https://hf-mirror.com/Veldrovive/upsamplers/resolve/main/working/latest.pth
echo 正在下载 second decoder 配置...
curl -L -o models\second_decoder_config.json https://hf-mirror.com/Veldrovive/upsamplers/raw/main/working/decoder_config.json

echo 正在下载 prior 模型...
curl -L -o models\prior.pth https://hf-mirror.com/laion/DALLE2-PyTorch/resolve/main/prior/latest.pth
echo 正在下载 prior 配置...
curl -L -o models\prior_config.json https://hf-mirror.com/laion/DALLE2-PyTorch/raw/main/prior/prior_config.json

echo ✅ 所有模型下载完成！
pause