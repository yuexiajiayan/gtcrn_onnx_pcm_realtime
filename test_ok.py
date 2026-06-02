import numpy as np
import onnxruntime


INPUT_PCM = "G:/test/16k.pcm"
OUTPUT_PCM = "C:/Users/lilin/Desktop/16k_1.pcm"
ONNX_MODEL = "onnx_models/gtcrn_simple.onnx"

N_FFT = 512
HOP_LEN = 256
READ_BYTES = 2560
INT16_MAX = 32768.0


def pcm_bytes_to_float(raw_data):
    pcm = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    return pcm / INT16_MAX


def float_to_pcm_bytes(samples):
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    return pcm.tobytes()


def enhance_frame(session, frame, window, conv_cache, tra_cache, inter_cache):
    spec = np.fft.rfft(frame * window, n=N_FFT).astype(np.complex64)
    mix = np.stack([spec.real, spec.imag], axis=-1)[None, :, None, :].astype(np.float32)

    enh, conv_cache, tra_cache, inter_cache = session.run(
        [],
        {
            "mix": mix,
            "conv_cache": conv_cache,
            "tra_cache": tra_cache,
            "inter_cache": inter_cache,
        },
    )

    enh = enh[0, :, 0, :]
    enhanced_frame = np.fft.irfft(enh[:, 0] + 1j * enh[:, 1], n=N_FFT).astype(np.float32)
    return enhanced_frame * window, conv_cache, tra_cache, inter_cache


def main():
    session = onnxruntime.InferenceSession(
        ONNX_MODEL,
        None,
        providers=["CPUExecutionProvider"],
    )

    window = np.sqrt(np.hanning(N_FFT)).astype(np.float32)
    conv_cache = np.zeros([2, 1, 16, 16, 33], dtype=np.float32)
    tra_cache = np.zeros([2, 3, 1, 1, 16], dtype=np.float32)
    inter_cache = np.zeros([2, 1, 33, 16], dtype=np.float32)

    input_cache = np.zeros(N_FFT, dtype=np.float32)
    output_cache = np.zeros(N_FFT, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)
    total_input_samples = 0
    total_output_samples = 0

    xxxx = open(OUTPUT_PCM, "wb")
    try:
        with open(INPUT_PCM, "rb") as f:
            while True:
                raw_data = f.read(READ_BYTES)
                if len(raw_data) == 0:
                    break

                samples = pcm_bytes_to_float(raw_data)
                total_input_samples += samples.shape[0]
                pending = np.concatenate([pending, samples])

                while pending.shape[0] >= HOP_LEN:
                    input_cache[:-HOP_LEN] = input_cache[HOP_LEN:]
                    input_cache[-HOP_LEN:] = pending[:HOP_LEN]
                    pending = pending[HOP_LEN:]

                    enhanced_frame, conv_cache, tra_cache, inter_cache = enhance_frame(
                        session,
                        input_cache,
                        window,
                        conv_cache,
                        tra_cache,
                        inter_cache,
                    )

                    output_cache += enhanced_frame
                    out = output_cache[:HOP_LEN]
                    xxxx.write(float_to_pcm_bytes(out))
                    total_output_samples += out.shape[0]

                    output_cache[:-HOP_LEN] = output_cache[HOP_LEN:]
                    output_cache[-HOP_LEN:] = 0.0

            if pending.shape[0] > 0:
                pending = np.pad(pending, (0, HOP_LEN - pending.shape[0]))
                input_cache[:-HOP_LEN] = input_cache[HOP_LEN:]
                input_cache[-HOP_LEN:] = pending[:HOP_LEN]

                enhanced_frame, conv_cache, tra_cache, inter_cache = enhance_frame(
                    session,
                    input_cache,
                    window,
                    conv_cache,
                    tra_cache,
                    inter_cache,
                )
                output_cache += enhanced_frame

                need = total_input_samples - total_output_samples
                if need > 0:
                    out = output_cache[:HOP_LEN]
                    xxxx.write(float_to_pcm_bytes(out[:need]))
                    total_output_samples += min(need, out.shape[0])
    finally:
        xxxx.close()

    print(f"done: {INPUT_PCM} -> {OUTPUT_PCM}, samples={total_output_samples}")


if __name__ == "__main__":
    main()
