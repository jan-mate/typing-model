import json
from pathlib import Path
from bokeh.io import output_notebook, show, save
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

PLOTS_DIR = Path(__file__).resolve().parents[2] / "plots"

MODELED_SLOTS = set("abcdefghijklmnopqrstuvwxyz0123456789 '.,/;")

def plot_keyboard(remap_path, layout_path='../data/layouts/layout_map.json', show_x_pos=False, color_fingers=False, draw_split_line=False, show_y_labels=False, modeled_only=False, save_html=True):
    output_notebook()

    with open(layout_path, 'r') as f:
        layout_data = json.load(f)
    
    with open(remap_path, 'r') as f:
        remap_data = json.load(f)

    layout_lookup = {item['slot']: item for item in layout_data}

    pastel_colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#E0E0E0"]
    finger_names = ["Pinky", "Ring", "Middle", "Index", "Thumb"]

    xs, ys, ws, hs, labels, colors, leg_labels = [], [], [], [], [], [], []
    left_max_x_per_y = {}
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for entry in remap_data:
        slot = entry['slot']
        keycode = entry['keycode']

        if slot not in layout_lookup:
            continue

        if modeled_only and str(slot).lower() not in MODELED_SLOTS:
            continue
            
        layout_info = layout_lookup[slot]
        raw_x = layout_info['x']
        y = layout_info['y']
        f_idx = layout_info['finger']
        
        display_f_idx = f_idx if f_idx < 5 else (9 - f_idx)
        
        if f_idx >= 5:
            display_x = 9.0 - raw_x
        else:
            display_x = raw_x
            
        if keycode == " ":
            width, height = 5.25, 0.90
            rect_x = 2.5 + (width / 2) 
            label = " "
            fill = "#E0E0E0"
        else:
            width, height = 0.90, 0.90
            rect_x = display_x + 0.5
            label = str(raw_x) if show_x_pos else keycode
            fill = pastel_colors[display_f_idx] if color_fingers else "#E0E0E0"

            if f_idx < 5:
                y_center = y + 0.5
                if y_center not in left_max_x_per_y or rect_x > left_max_x_per_y[y_center]:
                    left_max_x_per_y[y_center] = rect_x

        xs.append(rect_x)
        ys.append(y + 0.5)
        ws.append(width)
        hs.append(height)
        labels.append(label)
        colors.append(fill)
        leg_labels.append(finger_names[display_f_idx])
        
        min_x, max_x = min(min_x, rect_x - 0.5), max(max_x, rect_x + 0.5)
        min_y, max_y = min(min_y, y), max(max_y, y + 1)

    left_margin = 1.1 if show_y_labels else 0.65
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
        x=xs, y=ys, w=ws, h=hs, label=labels, color=colors, legend_label=leg_labels
    ))

    rect_kwargs = dict(
        x="x", y="y", width="w", height="h", source=source, 
        fill_color="color", line_color="#FFFFFF", line_width=0, 
        border_radius=6
    )
    if color_fingers:
        rect_kwargs["legend_group"] = "legend_label"

    p.rect(**rect_kwargs)

    if draw_split_line and left_max_x_per_y:
        sorted_ys = sorted(left_max_x_per_y.keys())
        line_xs, line_ys = [], []
        for i, cy in enumerate(sorted_ys):
            x_boundary = left_max_x_per_y[cy] + 0.5
            if i == 0:
                line_xs.extend([x_boundary, x_boundary])
                line_ys.extend([cy - 0.5, cy + 0.5])
            else:
                line_xs.extend([x_boundary, x_boundary])
                line_ys.extend([line_ys[-1], cy + 0.5])
        p.line(x=line_xs, y=line_ys, line_color="black", line_dash="dashed", line_width=2)

    if show_y_labels:
        centers = sorted(set(ys))
        lab_src = ColumnDataSource(data=dict(
            x=[min_x - 0.4] * len(centers),
            y=list(centers),
            t=[f"y={int(round(c - 0.5))}" for c in centers],
        ))
        p.text(
            x="x", y="y", text="t", source=lab_src,
            text_color="#000000", text_align="right",
            text_baseline="middle", text_font_size="20pt",
        )

    if color_fingers and p.legend:
        p.legend[0].items = sorted(p.legend[0].items, key=lambda item: finger_names.index(item.label['value']))
        p.legend[0].border_line_color = None
        p.legend[0].background_fill_alpha = 0.0
        p.legend[0].location = (12, 95)
        p.legend[0].label_text_font_size = "21pt"

    p.text(
        x="x", y="y", text="label", source=source, 
        text_color="#000000", text_align="center", 
        text_baseline="middle", text_font_size="24pt"
    )

    if save_html:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        name = Path(remap_path).stem
        parts = []
        if show_x_pos:
            parts.append("x")
        if color_fingers:
            parts.append("fingers")
        if draw_split_line:
            parts.append("split")
        if show_y_labels:
            parts.append("y")
        if modeled_only:
            parts.append("modeled")
        suffix = ("_" + "_".join(parts)) if parts else ""
        out_path = PLOTS_DIR / f"keyboard_{name}{suffix}.html"
        save(p, filename=str(out_path), title=f"keyboard_{name}{suffix}", resources="cdn")
        print(f"saved plot to {out_path}")

    try:
        show(p)
    except RuntimeError:
        # save() binds the model to a document, so showing the same model
        # afterward raises "Models must be owned by only a single document".
        # The saved HTML is the artifact, so this is non-fatal.
        pass