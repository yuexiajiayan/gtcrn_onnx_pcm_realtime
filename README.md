# gtcrn_onnx_pcm_realtime
用chatgpt ai适配流式pcm 降噪，使用模型16k：https://github.com/Xiaobin-Rong/gtcrn文件gtcrn_stream.py

# 需要环境
onnxruntime：https://github.com/microsoft/onnxruntime

kissfft: https://github.com/mborgerding/kissfft ( make KISSFFT_DATATYPE=float      KISSFFT_STATIC=1      KISSFFT_TOOLS=0      all)


# 编译运行cpp
g++ -O2 -std=c++17  -I./kissfft-master  -I./onnxruntime-linux/include  gtcrn_ok.cpp kissfft-master/libkissfft-float.a  -L./onnxruntime-linux/lib -lonnxruntime  -lm -lpthread  -o gtcrn_ok

./gtcrn_ok

# 结果和test_ok.py 运行一致，
# 可以用到jni环境



