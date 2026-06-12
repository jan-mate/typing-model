import json
from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, Title
from bokeh.layouts import gridplot
from src.utils.plot import MODELED_SLOTS

def create_layout_plot(remap_path, title_text):
    layout_path = 'data/layouts/layout_map.json'
    
    with open(layout_path, 'r') as f:
        layout_data = json.load(f)
        
    with open(remap_path, 'r') as f:
        remap_data = json.load(f)
        
    layout_lookup = {item['slot']: item for item in layout_data}

    xs, ys, ws, hs, labels, colors = [], [], [], [], [], []
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for entry in remap_data:
        slot = entry['slot']
        keycode = entry['keycode']
        
        if slot not in layout_lookup:
            continue

        if str(slot).lower() not in MODELED_SLOTS:
            continue

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
            fill = "#E0E0E0"
        else:
            width, height = 0.90, 0.90
            rect_x = display_x + 0.5
            label = keycode
            fill = "#E0E0E0"
            
        xs.append(rect_x)
        ys.append(y + 0.5)
        ws.append(width)
        hs.append(height)
        labels.append(label)
        colors.append(fill)
        
        min_x, max_x = min(min_x, rect_x - 0.5), max(max_x, rect_x + 0.5)
        min_y, max_y = min(min_y, y), max(max_y, y + 1)

    left_margin = 0.65
    x_span = (max_x + 0.65) - (min_x - left_margin)
    y_span = (max_y + 0.65) - (min_y - 0.65)
    plot_width = 800
    plot_height = max(1, round(plot_width * y_span / x_span))

    p = figure(
        x_range=(min_x - left_margin, max_x + 0.65),
        y_range=(min_y - 0.65, max_y + 0.65),
        width=plot_width,
        height=plot_height + 50, # extra space for title
        match_aspect=True,
        toolbar_location=None
    )

    p.title.text = title_text
    p.title.text_font_size = "20pt"
    p.title.align = "center"

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
           text_baseline="middle", text_font_size="16pt")
           
    return p

p_qwerty = create_layout_plot("data/layouts/qwerty_us.json", "US QWERTY")
p_dvorak = create_layout_plot("data/layouts/dvorak.json", "Dvorak")
p_colemak = create_layout_plot("data/layouts/colemak.json", "Colemak")
p_random = create_layout_plot("data/layouts/random.json", "Random")

grid = gridplot([[p_qwerty, p_dvorak], [p_colemak, p_random]], sizing_mode="scale_width")

out_path = "report/all_layouts.html"
save(grid, filename=out_path, title="All Layouts", resources="cdn")
print(f"saved grid plot to {out_path}")
