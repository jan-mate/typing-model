import pandas as pd
import os
from tqdm import tqdm

def get_edit_operations(target, typed):
    n, m = len(target), len(typed)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0], op[i][0] = i, ("D", i - 1, None)
    for j in range(1, m + 1): dp[0][j], op[0][j] = j, ("I", None, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == typed[j - 1]: cost, action = dp[i - 1][j - 1], ("M", i - 1, j - 1)
            else: cost, action = dp[i - 1][j - 1] + 1, ("S", i - 1, j - 1)
            if dp[i][j - 1] + 1 < cost: cost, action = dp[i][j - 1] + 1, ("I", None, j - 1)
            if dp[i - 1][j] + 1 < cost: cost, action = dp[i - 1][j] + 1, ("D", i - 1, None)
            if i >= 2 and j >= 2:
                if target[i-2] == typed[j-1] and target[i-1] == typed[j-2] and target[i-2] != target[i-1]:
                    if dp[i - 2][j - 2] + 1 < cost: cost, action = dp[i - 2][j - 2] + 1, ("T", i - 2, j - 2)
            dp[i][j], op[i][j] = cost, action
    actions, ci, cj = [], n, m
    while ci > 0 or cj > 0:
        a = op[ci][cj]
        if not a: break
        actions.append(a)
        if a[0] in ("M", "S"): ci, cj = ci - 1, cj - 1
        elif a[0] == "I": cj -= 1
        elif a[0] == "D": ci -= 1
        elif a[0] == "T": ci, cj = ci - 2, cj - 2
    return actions[::-1]

def reconstruct_final_text(group):
    stack, corrected = [], False
    for _, row in group.iterrows():
        c = str(row["LETTER"])
        if c == "SHIFT": continue
        if c == "BKSP":
            if stack:
                stack.pop()
                corrected = True
            continue
        item = row.to_dict()
        item["_CORR"] = corrected
        stack.append(item)
        corrected = False
    return stack

def label_keystrokes(pid, sid, stack, target):
    typed = [str(x["LETTER"]) for x in stack]
    acts = get_edit_operations(list(target), typed)
    lbls = [{"is": False, "t": None} for _ in range(len(stack))]
    for tag, _, tj in acts:
        if tag == "S": lbls[tj] = {"is": True, "t": "Substitution"}
        elif tag == "I": lbls[tj] = {"is": True, "t": "Insertion"}
        elif tag == "T":
            lbls[tj] = {"is": True, "t": "Transposition"}
            if tj + 1 < len(lbls): lbls[tj+1] = {"is": True, "t": "Transposition"}
    for k, (tag, _, tj) in enumerate(acts):
        if tag == "D":
            p_idx = next((a[2] for a in reversed(acts[:k]) if a[2] is not None), None)
            n_idx = next((a[2] for a in acts[k+1:] if a[2] is not None), None)
            for i in [p_idx, n_idx]:
                if i is not None and 0 <= i < len(lbls) and not lbls[i]["is"]:
                    lbls[i] = {"is": True, "t": "Deletion"}
    for i, item in enumerate(stack):
        if item.get("_CORR"): lbls[i] = {"is": True, "t": "Proofreading"}
    t0 = stack[0]["PRESS_TIME"]
    return [{
        "PARTICIPANT_ID": pid, "SEQUENCE_ID": f"{pid}_{sid}", 
        "ORIGINAL_SEQUENCE_ID": f"{pid}_{sid}", "KEY": str(stack[i]["LETTER"]),
        "TIME": int(stack[i]["PRESS_TIME"] - t0), "IS_TYPO": lbls[i]["is"], "TYPO_TYPE": lbls[i]["t"]
    } for i in range(len(stack))]

def run(participant_file, keystrokes_dir, output_path):
    p_ids = pd.read_csv(participant_file, sep="\t")["PARTICIPANT_ID"].tolist()
    rows = []
    for pid in tqdm(p_ids, desc="Annotating Keystrokes"):
        path = os.path.join(keystrokes_dir, f"{pid}_keystrokes.txt")
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path, sep="\t", quoting=3, on_bad_lines="skip", escapechar="\\")
            df = df.dropna(subset=["PRESS_TIME", "LETTER", "TEST_SECTION_ID"])
        except: continue
        for sid, group in df.groupby("TEST_SECTION_ID"):
            try:
                clean_sid = int(float(sid))
                group = group.sort_values("PRESS_TIME").reset_index(drop=True)
                stack = reconstruct_final_text(group)
                if not stack: continue
                rows.extend(label_keystrokes(pid, clean_sid, stack, str(group["SENTENCE"].iloc[0])))
            except: continue
    final_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path)
    
if __name__ == "__main__":
    run("data/interim/participants/qwerty_80_us_9-10.txt", "data/raw/files/", "data/interim/annotated_sequences.parquet")