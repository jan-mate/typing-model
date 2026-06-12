import json
from pathlib import Path
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource

def plot_modeled_keyboard():
    layout_path = 'data/layouts/layout_map.json'
    
    with open(layout_path, 'r') as f:
        layout_data = json.load(f)
        
    modeled_keys = set("abcdefghijklmnopqrstuvwxyz0123456789 '.,/;")
    
    remap_data = [item for item in layout_data if str(item['slot']).lower() in modeled_keys]
    layout_lookup = {item['slot']: item for item in layout_data}

    pastel_colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#E0E0E0"]
    finger_names = ["Pinky", "Ring", "Middle", "Index", "Thumb"]

    xs, ys, ws, hs, labels, colors = [], [], [], [], [], []
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for entry in remap_data:
        slot = entry['slot']
        keycode = slot 
        
        layout_info = layout_lookup[slot]
        raw_x = layout_info['x']
        y = layout_info['y']
        f_idx = layout_info['finger']
        
        display_f_idx = f_idx if f_idx < 5 else (9 - f_idx)
        display_x = 9.0 - raw_x if f_idx >= 5 else raw_x
            
        if keycode == " ":
            width, height = 5.25, 0.90
            rect_x = 2.5 + (width / 2) 
            label = " "
        else:
            width, height = 0.90, 0.90
            rect_x = display_x + 0.5
            label = keycode
            
        xs.append(rect_x)
        ys.append(y + 0.5)
        ws.append(width)
        hs.append(height)
        labels.append(label)
        colors.append("#E0E0E0")
        
        min_x, max_x = min(min_x, rect_x - 0.5), max(max_x, rect_x + 0.5)
        min_y, max_y = min(min_y, y), max(max_y, y + 1)

    left_margin = 0.65
    x_span = (max_x + 0.65) - (min_x - left_margin)
    y_span = (max_y + 0.65) - (min_y - 0.65)
    plot_width = 1350
    plot_height = max(1, round(plot_width * y_span / x_span))

    p = figure(
        x_range=(min_x - left_margin, max_x + 0.65),
        y_range=(min_y - 0.65, max_y + 0.65),
        width=plot_width,
        height=plot_height,
        match_aspect=True,
        toolbar_location=None
    )

    p.outline_line_color = None
    p.axis.visible = False
    p.grid.visible = False

    source = ColumnDataSource(data=dict(
        x=xs, y=ys, w=ws, h=hs, label=labels, color=colors
    ))

    p.rect(x="x", y="y", width="w", height="h", source=source, 
           fill_color="color", line_color="#FFFFFF", line_width=0, 
           border_radius=6)

    p.text(x="x", y="y", text="label", source=source, 
           text_color="#000000", text_align="center", 
           text_baseline="middle", text_font_size="24pt")

    out_path = "report/modeled_keys.html"
    save(p, filename=out_path, title="modeled_keys", resources="cdn")
    print(f"saved plot to {out_path}")

plot_modeled_keyboard()
