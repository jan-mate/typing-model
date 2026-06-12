import json
import numpy as np
import os

def calculate_movement_features(layout_map_path, output_path):
    with open(layout_map_path, 'r') as f:
        layout_map = json.load(f)

    # ordered fallbacks: try the QWERTY home slot first; if it's been remapped
    # (e.g. ";" replaced by the repeat key "\x02" in rpt_trg layouts), fall
    # back to the next candidate so home_coords stays defined
    finger_home_fallbacks = {
        0: ["a"], 1: ["s"], 2: ["d"], 3: ["f"],
        4: [" "], 5: [" "],
        6: ["j"], 7: ["k"], 8: ["l"], 9: [";", "\x02"],
    }

    slot_lookup = {item['slot']: item for item in layout_map}
    home_coords = {}
    for finger, fallbacks in finger_home_fallbacks.items():
        for s_name in fallbacks:
            if s_name in slot_lookup:
                home_coords[finger] = (slot_lookup[s_name]['x'], slot_lookup[s_name]['y'])
                break

    reach_data = {}
    for s_id, info in slot_lookup.items():
        f_id = info['finger']
        if f_id in home_coords:
            hx, hy = home_coords[f_id]
            dx, dy = info['x'] - hx, info['y'] - hy
            dist = float(np.hypot(dx, dy))
            if dist > 0:
                angle = np.arctan2(dy, dx)
                reach_data[s_id] =[dist, float(np.sin(angle)), float(np.cos(angle))]
            else:
                reach_data[s_id] =[0.0, -2.0, -2.0]
        else:
            reach_data[s_id] =[0.0, -2.0, -2.0]

    bigram_data = {}
    slots = list(slot_lookup.keys())
    for s1 in slots:
        for s2 in slots:
            info1, info2 = slot_lookup[s1], slot_lookup[s2]
            
            if info1['finger'] == info2['finger']:
                dx, dy = info2['x'] - info1['x'], info2['y'] - info1['y']
                dist = float(np.hypot(dx, dy))

                if dist > 0:
                    angle = np.arctan2(dy, dx)
                    bigram_data[f"{s1}{s2}"] =[dist, float(np.sin(angle)), float(np.cos(angle))]
                else:
                    bigram_data[f"{s1}{s2}"] = [0.0, -2.0, -2.0]
                    
        
            else:
                bigram_data[f"{s1}{s2}"] = reach_data[s2]

    output_data = {"reach": reach_data, "bigram": bigram_data}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    calculate_movement_features('data/layouts/layout_map.json', 'data/layouts/movement_features.json')