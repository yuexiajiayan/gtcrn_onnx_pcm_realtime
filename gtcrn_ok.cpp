#include <onnxruntime_cxx_api.h>
#include "kiss_fftr.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* INPUT_PCM = "16k.pcm";
constexpr const char* OUTPUT_PCM = "16k_2.pcm";
#ifdef _WIN32
constexpr const wchar_t* ONNX_MODEL = L"gtcrn_simple.onnx";
#else
constexpr const char* ONNX_MODEL = "gtcrn_simple.onnx";
#endif

constexpr int N_FFT = 512;
constexpr int FREQ_BINS = N_FFT / 2 + 1;
constexpr int HOP_LEN = 256;
constexpr int READ_BYTES = 2560;
constexpr float INT16_MAX_FLOAT = 32768.0f;
constexpr float INT16_WRITE_SCALE = 32767.0f;
constexpr float PI = 3.14159265358979323846f;

constexpr size_t MIX_SIZE = 1 * FREQ_BINS * 1 * 2;
constexpr size_t CONV_CACHE_SIZE = 2 * 1 * 16 * 16 * 33;
constexpr size_t TRA_CACHE_SIZE = 2 * 3 * 1 * 1 * 16;
constexpr size_t INTER_CACHE_SIZE = 2 * 1 * 33 * 16;

std::vector<float> BuildSqrtHannWindow() {
    std::vector<float> window(N_FFT);
    for (int i = 0; i < N_FFT; ++i) {
        const float hann = 0.5f - 0.5f * std::cos(2.0f * PI * static_cast<float>(i) / static_cast<float>(N_FFT - 1));
        window[i] = std::sqrt(std::max(0.0f, hann));
    }
    return window;
}

std::vector<float> PcmBytesToFloat(const std::vector<char>& raw_data, std::streamsize bytes_read) {
    const size_t sample_count = static_cast<size_t>(bytes_read) / sizeof(int16_t);
    std::vector<float> samples(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        int16_t sample = 0;
        std::memcpy(&sample, raw_data.data() + i * sizeof(int16_t), sizeof(int16_t));
        samples[i] = static_cast<float>(sample) / INT16_MAX_FLOAT;
    }
    return samples;
}

int16_t FloatToPcmSample(float sample) {
    sample = std::clamp(sample, -1.0f, 1.0f);
    return static_cast<int16_t>(sample * INT16_WRITE_SCALE);
}

void WritePcmSamples(std::ofstream& output, const float* samples, size_t count) {
    std::vector<int16_t> pcm(count);
    for (size_t i = 0; i < count; ++i) {
        pcm[i] = FloatToPcmSample(samples[i]);
    }
    output.write(reinterpret_cast<const char*>(pcm.data()), static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
}

std::vector<float> RunFrame(
    Ort::Session& session,
    Ort::MemoryInfo& memory_info,
    kiss_fftr_cfg fft_cfg,
    kiss_fftr_cfg ifft_cfg,
    const std::vector<float>& input_cache,
    const std::vector<float>& window,
    std::vector<float>& mix,
    std::vector<float>& conv_cache,
    std::vector<float>& tra_cache,
    std::vector<float>& inter_cache) {

    std::array<kiss_fft_scalar, N_FFT> fft_input{};
    std::array<kiss_fft_cpx, FREQ_BINS> fft_output{};

    for (int i = 0; i < N_FFT; ++i) {
        fft_input[i] = input_cache[i] * window[i];
    }

    kiss_fftr(fft_cfg, fft_input.data(), fft_output.data());

    for (int i = 0; i < FREQ_BINS; ++i) {
        mix[i * 2] = fft_output[i].r;
        mix[i * 2 + 1] = fft_output[i].i;
    }

    std::array<int64_t, 4> mix_shape{1, FREQ_BINS, 1, 2};
    std::array<int64_t, 5> conv_shape{2, 1, 16, 16, 33};
    std::array<int64_t, 5> tra_shape{2, 3, 1, 1, 16};
    std::array<int64_t, 4> inter_shape{2, 1, 33, 16};

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, mix.data(), mix.size(), mix_shape.data(), mix_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, conv_cache.data(), conv_cache.size(), conv_shape.data(), conv_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, tra_cache.data(), tra_cache.size(), tra_shape.data(), tra_shape.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, inter_cache.data(), inter_cache.size(), inter_shape.data(), inter_shape.size()));

    const char* input_names[] = {"mix", "conv_cache", "tra_cache", "inter_cache"};
    const char* output_names[] = {"enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        4);

    const float* enh = output_tensors[0].GetTensorData<float>();
    const float* conv_out = output_tensors[1].GetTensorData<float>();
    const float* tra_out = output_tensors[2].GetTensorData<float>();
    const float* inter_out = output_tensors[3].GetTensorData<float>();

    std::copy(conv_out, conv_out + conv_cache.size(), conv_cache.begin());
    std::copy(tra_out, tra_out + tra_cache.size(), tra_cache.begin());
    std::copy(inter_out, inter_out + inter_cache.size(), inter_cache.begin());

    std::array<kiss_fft_cpx, FREQ_BINS> ifft_input{};
    std::array<kiss_fft_scalar, N_FFT> ifft_output{};

    for (int i = 0; i < FREQ_BINS; ++i) {
        ifft_input[i].r = enh[i * 2];
        ifft_input[i].i = enh[i * 2 + 1];
    }

    kiss_fftri(ifft_cfg, ifft_input.data(), ifft_output.data());

    std::vector<float> enhanced_frame(N_FFT);
    for (int i = 0; i < N_FFT; ++i) {
        enhanced_frame[i] = ifft_output[i] / static_cast<float>(N_FFT) * window[i];
    }
    return enhanced_frame;
}

void ProcessOneHop(
    Ort::Session& session,
    Ort::MemoryInfo& memory_info,
    kiss_fftr_cfg fft_cfg,
    kiss_fftr_cfg ifft_cfg,
    std::deque<float>& pending,
    std::vector<float>& input_cache,
    std::vector<float>& output_cache,
    const std::vector<float>& window,
    std::vector<float>& mix,
    std::vector<float>& conv_cache,
    std::vector<float>& tra_cache,
    std::vector<float>& inter_cache,
    std::ofstream& output,
    int64_t& total_output_samples) {

    std::copy(input_cache.begin() + HOP_LEN, input_cache.end(), input_cache.begin());
    for (int i = 0; i < HOP_LEN; ++i) {
        input_cache[N_FFT - HOP_LEN + i] = pending.front();
        pending.pop_front();
    }

    const std::vector<float> enhanced_frame = RunFrame(
        session,
        memory_info,
        fft_cfg,
        ifft_cfg,
        input_cache,
        window,
        mix,
        conv_cache,
        tra_cache,
        inter_cache);

    for (int i = 0; i < N_FFT; ++i) {
        output_cache[i] += enhanced_frame[i];
    }

    WritePcmSamples(output, output_cache.data(), HOP_LEN);
    total_output_samples += HOP_LEN;

    std::copy(output_cache.begin() + HOP_LEN, output_cache.end(), output_cache.begin());
    std::fill(output_cache.end() - HOP_LEN, output_cache.end(), 0.0f);
}

}  // namespace

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gtcrn_ok");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, ONNX_MODEL, session_options);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        using KissFftrPtr = std::unique_ptr<kiss_fftr_state, decltype(&kiss_fftr_free)>;
        KissFftrPtr fft_cfg(kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr), kiss_fftr_free);
        KissFftrPtr ifft_cfg(kiss_fftr_alloc(N_FFT, 1, nullptr, nullptr), kiss_fftr_free);
        if (!fft_cfg || !ifft_cfg) {
            throw std::runtime_error("kiss_fftr_alloc failed");
        }

        const std::vector<float> window = BuildSqrtHannWindow();
        std::vector<float> mix(MIX_SIZE, 0.0f);
        std::vector<float> conv_cache(CONV_CACHE_SIZE, 0.0f);
        std::vector<float> tra_cache(TRA_CACHE_SIZE, 0.0f);
        std::vector<float> inter_cache(INTER_CACHE_SIZE, 0.0f);
        std::vector<float> input_cache(N_FFT, 0.0f);
        std::vector<float> output_cache(N_FFT, 0.0f);
        std::deque<float> pending;

        std::ifstream input(INPUT_PCM, std::ios::binary);
        if (!input) {
            throw std::runtime_error(std::string("failed to open input: ") + INPUT_PCM);
        }

        std::ofstream output(OUTPUT_PCM, std::ios::binary);
        if (!output) {
            throw std::runtime_error(std::string("failed to open output: ") + OUTPUT_PCM);
        }

        std::vector<char> raw_data(READ_BYTES);
        int64_t total_input_samples = 0;
        int64_t total_output_samples = 0;

        while (true) {
            input.read(raw_data.data(), raw_data.size());
            const std::streamsize bytes_read = input.gcount();
            if (bytes_read == 0) {
                break;
            }

            const std::vector<float> samples = PcmBytesToFloat(raw_data, bytes_read);
            total_input_samples += static_cast<int64_t>(samples.size());
            for (float sample : samples) {
                pending.push_back(sample);
            }

            while (pending.size() >= HOP_LEN) {
                ProcessOneHop(
                    session,
                    memory_info,
                    fft_cfg.get(),
                    ifft_cfg.get(),
                    pending,
                    input_cache,
                    output_cache,
                    window,
                    mix,
                    conv_cache,
                    tra_cache,
                    inter_cache,
                    output,
                    total_output_samples);
            }
        }

        if (!pending.empty()) {
            while (pending.size() < HOP_LEN) {
                pending.push_back(0.0f);
            }

            std::copy(input_cache.begin() + HOP_LEN, input_cache.end(), input_cache.begin());
            for (int i = 0; i < HOP_LEN; ++i) {
                input_cache[N_FFT - HOP_LEN + i] = pending.front();
                pending.pop_front();
            }

            const std::vector<float> enhanced_frame = RunFrame(
                session,
                memory_info,
                fft_cfg.get(),
                ifft_cfg.get(),
                input_cache,
                window,
                mix,
                conv_cache,
                tra_cache,
                inter_cache);

            for (int i = 0; i < N_FFT; ++i) {
                output_cache[i] += enhanced_frame[i];
            }

            const int64_t need = total_input_samples - total_output_samples;
            if (need > 0) {
                const size_t write_count = static_cast<size_t>(std::min<int64_t>(need, HOP_LEN));
                WritePcmSamples(output, output_cache.data(), write_count);
                total_output_samples += static_cast<int64_t>(write_count);
            }
        }

        std::cout << "done: " << INPUT_PCM << " -> " << OUTPUT_PCM
                  << ", samples=" << total_output_samples << std::endl;
        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "onnxruntime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
}
