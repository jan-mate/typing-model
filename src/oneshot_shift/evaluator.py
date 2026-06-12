import numpy as np
import pandas as pd
from src.utils.inference import predict_speed

IKI_MEAN_MS = 110.66
IKI_MEDIAN_MS = 100.00
IKI_STD_MS = 52.40

class SpeedEvaluator:
    def __init__(self, model, base_engine, oss_engine, baseline_layout, oss_layout):
        self.model = model
        self.base_engine = base_engine  # regular frequency tables
        self.oss_engine = oss_engine    # 1SS-aware frequency tables
        self.baseline_layout = baseline_layout
        self.oss_layout = oss_layout

    def evaluate(self, sentences: list, oss_transformer):
        print(f"Batching inference for {len(sentences)} sentences...")

        oss_texts = [oss_transformer.transform(s) for s in sentences]
        oss_char = oss_transformer.one_shot_char

        # a sentence-initial capital's leading 1SS letter is free, like the standard cap
        skip_second = {i for i, t in enumerate(oss_texts) if t.startswith(oss_char)}

        print("Predicting baseline costs...")
        base_results = predict_speed(sentences, self.model, self.base_engine, self.baseline_layout)

        print("Predicting 1SS costs...")
        oss_results = predict_speed(oss_texts, self.model, self.oss_engine, self.oss_layout, skip_second=skip_second)

        data = []
        for i in range(len(sentences)):
            n_oss_keys = oss_texts[i].count(oss_transformer.one_shot_char)
            oss_n = len(oss_texts[i]) - (1 if i in skip_second else 0)

            t_base_mean = self._calculate_ms(len(sentences[i]), base_results[i][0], IKI_MEAN_MS)
            t_oss_mean = self._calculate_ms(oss_n, oss_results[i][0], IKI_MEAN_MS)

            # median as the baseline per-key cost
            t_base_median = self._calculate_ms(len(sentences[i]), base_results[i][0], IKI_MEDIAN_MS)
            t_oss_median = self._calculate_ms(oss_n, oss_results[i][0], IKI_MEDIAN_MS)

            # motion-only: the 1SS key itself takes 0ms
            t_oss_motion = t_oss_mean - (n_oss_keys * IKI_MEAN_MS)

            data.append({
                "original": sentences[i],
                "transformed": oss_texts[i],
                "time_base_mean": t_base_mean,
                "time_base_median": t_base_median,
                "time_oss_mean": t_oss_mean,
                "time_oss_median": t_oss_median,
                "time_oss_motion": t_oss_motion,
                "savings_mean": t_base_mean - t_oss_mean,
                "savings_median": t_base_median - t_oss_median,
                "savings_motion": t_base_mean - t_oss_motion
            })
            
        return pd.DataFrame(data)

    def _calculate_ms(self, n_chars, sum_z, baseline):
        return max(0, n_chars - 1) * baseline + sum_z * IKI_STD_MS
