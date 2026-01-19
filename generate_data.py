import pandas as pd
import numpy as np
import random

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
# ==========================================
# Set seed for reproducibility (ensures you get the same numbers every time you run this)
np.random.seed(42)
random.seed(42)

# Total number of clinical consultation sessions to simulate
NUM_SAMPLES = 1000

def generate_medical_scribe_dataset():
    """
    Generates a synthetic dataset representing the operational performance 
    of the HM-3 AI Medical Scribe system in Malaysian clinical settings.
    """
    print(f"Generating {NUM_SAMPLES} simulated clinical sessions...")

    # ------------------------------------------
    # 1. INDEPENDENT VARIABLES (The Inputs)
    # ------------------------------------------
    
    # Session IDs
    session_ids = [f"SES-{1000+i}" for i in range(NUM_SAMPLES)]

    # Ambient Noise (dB): Normal distribution centered on typical hospital noise (55dB)
    # Range clipped to 35dB (Quiet room) to 85dB (Crowded ER)
    ambient_noise = np.random.normal(loc=55, scale=12, size=NUM_SAMPLES)
    ambient_noise = np.clip(ambient_noise, 35, 85)

    # Rojak Index (1-10): Represents linguistic complexity
    # 1 = Pure English/Malay, 10 = Heavy code-switching (Manglish + Dialects)
    rojak_index = np.random.randint(1, 11, size=NUM_SAMPLES)

    # Audio Duration (seconds): Length of doctor-patient consultation
    # Mean = 5 minutes (300s), Std Dev = 2 mins
    audio_duration = np.random.normal(loc=300, scale=120, size=NUM_SAMPLES)
    audio_duration = np.clip(audio_duration, 60, 900) # Min 1 min, Max 15 mins

    # Model Type: 0 = Standard Whisper Base, 1 = Fine-Tuned Local Model (HM-3)
    # We assume 50% of tests were done on the new model
    model_type_binary = np.random.randint(0, 2, size=NUM_SAMPLES)
    model_labels = ["Base_Model" if x == 0 else "HM3_FineTuned" for x in model_type_binary]

    # ------------------------------------------
    # 2. DEPENDENT VARIABLES (The Engineering Outcomes)
    # ------------------------------------------
    # We simulate "Physics" here: Noise and Complexity should degrade performance.
    
    # A. Processing Latency (seconds)
    # Logic: Longer audio takes longer to process. 
    # High Rojak Index adds computational overhead (simulating language switching delay).
    # The HM3_FineTuned model is slightly slower due to extra processing layers.
    base_processing_speed = 0.15  # It takes 0.15s to process 1s of audio
    latency = (audio_duration * base_processing_speed) + \
              (rojak_index * 1.5) + \
              (model_type_binary * 2.0) + \
              np.random.normal(0, 2, NUM_SAMPLES) # Add random jitter
    latency = np.clip(latency, 5, None) # Minimum 5 seconds latency

    # B. Word Error Rate (WER) %
    # Logic: This is the critical metric. 
    # - High Noise increases error significantly.
    # - High Rojak Index increases error significantly.
    # - The HM3_FineTuned model (Value 1) should REDUCE error by a large factor.
    wer = 5.0 + \
          (ambient_noise * 0.4) + \
          (rojak_index * 3.5) - \
          (model_type_binary * 15.0) + \
          np.random.normal(0, 3, NUM_SAMPLES)
    
    # Clip WER to realistic bounds (0% to 100%)
    wer = np.clip(wer, 2.0, 99.0)

    # ------------------------------------------
    # 3. DATAFRAME CREATION
    # ------------------------------------------
    df = pd.DataFrame({
        "Session_ID": session_ids,
        "Model_Type": model_labels,
        "Audio_Duration_Sec": np.round(audio_duration, 1),
        "Ambient_Noise_dB": np.round(ambient_noise, 1),
        "Rojak_Index": rojak_index,
        "Processing_Latency_Sec": np.round(latency, 2),
        "Word_Error_Rate_Pct": np.round(wer, 2)
    })

    # ------------------------------------------
    # 4. EXPORT
    # ------------------------------------------
    output_filename = "HM3_Scribe_Operational_Data.csv"
    df.to_csv(output_filename, index=False)
    
    print(f"\n[SUCCESS] Dataset generated: {output_filename}")
    print("-" * 50)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("-" * 50)
    print("\nStatistical Summary:")
    print(df.describe())

if __name__ == "__main__":
    generate_medical_scribe_dataset()