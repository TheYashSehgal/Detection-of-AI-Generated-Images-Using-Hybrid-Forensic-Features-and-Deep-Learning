import numpy as np

def compute_prob(ela, srm, dct, var, noise, corr):
    ela_ai_score = float(np.clip(1.0 - ela / 0.05, 0.0, 1.0))
    srm_ai_score = float(np.clip(1.0 - srm / 0.07, 0.0, 1.0))
    dct_ai_score = float(np.clip(dct, 0.0, 1.0))
    local_var_score = float(np.clip(var / 200.0, 0.0, 1.0))
    grad_noise_score = float(np.clip(noise / 20.0, 0.0, 1.0))
    corr_ai_score = float(np.clip(corr, 0.0, 1.0))

    raw = (
        0.30 * ela_ai_score +
        0.20 * srm_ai_score +
        0.15 * dct_ai_score +
        0.15 * (1.0 - local_var_score) +
        0.10 * (1.0 - grad_noise_score) +
        0.10 * corr_ai_score
    )
    raw = float(np.clip(raw, 0.01, 0.99))
    
    # Sharpness centered at 0.70
    sharpened = 1.0 / (1.0 + np.exp(-10.0 * (raw - 0.70)))
    return raw, sharpened

print("--- Typical Smartphone Photo (Real) ---")
# Somewhat smooth (low var/noise), high correlation, but HIGH ELA/SRM
raw, prob = compute_prob(ela=0.03, srm=0.04, dct=0.3, var=50, noise=5, corr=0.9)
print(f"Raw: {raw:.3f}, Prob: {prob:.3f}")

print("--- Typical Midjourney Image (AI) ---")
# Very smooth, perfect correlation, extremely low ELA/SRM
raw, prob = compute_prob(ela=0.002, srm=0.005, dct=0.8, var=20, noise=2, corr=0.98)
print(f"Raw: {raw:.3f}, Prob: {prob:.3f}")

print("--- Noisy Real Photo (Real) ---")
raw, prob = compute_prob(ela=0.06, srm=0.08, dct=0.1, var=150, noise=15, corr=0.6)
print(f"Raw: {raw:.3f}, Prob: {prob:.3f}")

print("--- AI Upscaled Photo (AI/Borderline) ---")
raw, prob = compute_prob(ela=0.01, srm=0.02, dct=0.6, var=80, noise=8, corr=0.85)
print(f"Raw: {raw:.3f}, Prob: {prob:.3f}")
