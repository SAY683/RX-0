import dash
from dash import dcc, html, State  # dcc (Dash Core Components), html (Dash HTML Components)
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import random
import uuid
import math  # For isnan

# --- 配置 ---
INITIAL_VOLUME = 1.0
INITIAL_EFFECT = 1.0


# --- 参数函数 (更复杂且可通过GUI控制) ---
def get_proportions(depth, parent_type, p_eff_base, eff_conc_factor, ineff_spread_factor, rand_factor):
    p_eff = p_eff_base

    if parent_type == "effective":  # 有效分支
        p_eff = max(0.01, p_eff_base - depth * eff_conc_factor * 0.01 - random.uniform(0, rand_factor * 0.05))
    elif parent_type == "ineffective":  # 无效分支
        p_eff = min(0.99, p_eff_base + depth * ineff_spread_factor * 0.01 + random.uniform(0, rand_factor * 0.05))

    p_eff = max(0.01, min(0.99, p_eff))
    return p_eff, 1.0 - p_eff


def get_effect_weights(depth, parent_type, w_eff_base, eff_effect_conc, ineff_effect_damp, rand_factor):
    w_eff = w_eff_base

    if parent_type == "effective":  # 有效分支
        w_eff = min(0.99, w_eff_base + depth * eff_effect_conc * 0.01 + random.uniform(0, rand_factor * 0.05))
    elif parent_type == "ineffective":  # 无效分支
        w_eff = max(0.01, w_eff_base - depth * ineff_effect_damp * 0.01 - random.uniform(0, rand_factor * 0.05))

    w_eff = max(0.01, min(0.99, w_eff))
    return w_eff, 1.0 - w_eff


# --- 数据生成函数 ---
def generate_fractal_data(max_depth,
                          p_eff_base, eff_conc_factor, ineff_spread_factor,
                          w_eff_base, eff_effect_conc, ineff_effect_damp,
                          rand_factor,
                          current_time_step=0):
    nodes_data = {
        "ids": [], "labels": [], "parents": [], "volumes": [],
        "effects": [], "types": [], "depths": [], "effect_densities": []
    }
    node_counter = 0

    def _recursive_generator(parent_id, parent_volume, parent_effect, parent_type_str, current_depth):
        nonlocal node_counter
        if current_depth > max_depth:
            return

        node_counter += 1

        prop_eff, prop_ineff = get_proportions(current_depth, parent_type_str, p_eff_base, eff_conc_factor,
                                               ineff_spread_factor, rand_factor)
        weight_eff, weight_ineff = get_effect_weights(current_depth, parent_type_str, w_eff_base, eff_effect_conc,
                                                      ineff_effect_damp, rand_factor)

        # --- 有效子节点 ---
        eff_id = f"节点_{node_counter}_{uuid.uuid4().hex[:4]}"
        eff_label = f"L{current_depth}-有效 (体积:{prop_eff * 100:.0f}%, 效应:{weight_eff * 100:.0f}%)"
        eff_volume = parent_volume * prop_eff
        eff_effect = parent_effect * weight_eff
        eff_density = (eff_effect / eff_volume) if eff_volume > 0 else 0

        nodes_data["ids"].append(eff_id)
        nodes_data["labels"].append(eff_label)
        nodes_data["parents"].append(parent_id)
        nodes_data["volumes"].append(eff_volume)
        nodes_data["effects"].append(eff_effect)
        nodes_data["types"].append("effective")  # 保持英文以便颜色映射
        nodes_data["depths"].append(current_depth)
        nodes_data["effect_densities"].append(eff_density if not math.isnan(eff_density) else 0)

        _recursive_generator(eff_id, eff_volume, eff_effect, "effective", current_depth + 1)

        # --- 无效子节点 ---
        node_counter += 1
        ineff_id = f"节点_{node_counter}_{uuid.uuid4().hex[:4]}"
        ineff_label = f"L{current_depth}-无效 (体积:{prop_ineff * 100:.0f}%, 效应:{weight_ineff * 100:.0f}%)"
        ineff_volume = parent_volume * prop_ineff
        ineff_effect = parent_effect * weight_ineff
        ineff_density = (ineff_effect / ineff_volume) if ineff_volume > 0 else 0

        nodes_data["ids"].append(ineff_id)
        nodes_data["labels"].append(ineff_label)
        nodes_data["parents"].append(parent_id)
        nodes_data["volumes"].append(ineff_volume)
        nodes_data["effects"].append(ineff_effect)
        nodes_data["types"].append("ineffective")  # 保持英文以便颜色映射
        nodes_data["depths"].append(current_depth)
        nodes_data["effect_densities"].append(ineff_density if not math.isnan(ineff_density) else 0)

        _recursive_generator(ineff_id, ineff_volume, ineff_effect, "ineffective", current_depth + 1)

    # 初始化系统
    root_id = "系统根节点"
    nodes_data["ids"].append(root_id)
    nodes_data["labels"].append(f"系统根节点 (时间步: {current_time_step})")
    nodes_data["parents"].append("")
    nodes_data["volumes"].append(INITIAL_VOLUME)
    nodes_data["effects"].append(INITIAL_EFFECT)
    nodes_data["types"].append("root")  # 保持英文以便颜色映射
    nodes_data["depths"].append(0)
    nodes_data["effect_densities"].append((INITIAL_EFFECT / INITIAL_VOLUME) if INITIAL_VOLUME > 0 else 0)

    _recursive_generator(root_id, INITIAL_VOLUME, INITIAL_EFFECT, "root", 1)
    return nodes_data


# --- Dash 应用设置 ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server

app.layout = html.Div([
    html.H1("多重分形系统浏览器 (二八定律与自相似性)"),

    html.Div([
        # --- 控制列 ---
        html.Div([
            html.H4("系统参数"),
            html.Label("最大深度:"),
            dcc.Slider(id='max-depth-slider', min=1, max=5, step=1, value=3, marks={i: str(i) for i in range(1, 6)}),

            html.Label("基础'有效'部分体积比例 (p_eff_base, %):"),
            dcc.Slider(id='p-eff-base-slider', min=5, max=50, step=1, value=20,
                       marks={i * 5: str(i * 5) for i in range(1, 11)}),

            html.Label("有效分支体积集中因子 (eff_conc_factor, %):"),
            dcc.Slider(id='eff-conc-factor-slider', min=0, max=10, step=0.5, value=2,
                       marks={i: str(i) for i in range(0, 11, 2)}),

            html.Label("无效分支体积扩散因子 (ineff_spread_factor, %):"),
            dcc.Slider(id='ineff-spread-factor-slider', min=0, max=10, step=0.5, value=1,
                       marks={i: str(i) for i in range(0, 11, 2)}),

            html.Label("基础'有效'部分效应权重 (w_eff_base, %):"),
            dcc.Slider(id='w-eff-base-slider', min=50, max=95, step=1, value=80,
                       marks={i * 5: str(i * 5) for i in range(10, 20)}),

            html.Label("有效分支效应集中因子 (eff_effect_conc, %):"),
            dcc.Slider(id='eff-effect-conc-slider', min=0, max=10, step=0.5, value=2,
                       marks={i: str(i) for i in range(0, 11, 2)}),

            html.Label("无效分支效应抑制因子 (ineff_effect_damp, %):"),
            dcc.Slider(id='ineff-effect-damp-slider', min=0, max=10, step=0.5, value=1,
                       marks={i: str(i) for i in range(0, 11, 2)}),

            html.Label("随机性因子 (rand_factor, %):"),
            dcc.Slider(id='rand-factor-slider', min=0, max=10, step=0.5, value=1,
                       marks={i: str(i) for i in range(0, 11, 2)}),

            html.Label("可视化类型:"),
            dcc.Dropdown(
                id='viz-type-dropdown',
                options=[
                    {'label': '旭日图 (按体积)', 'value': 'sunburst_volume'},
                    {'label': '旭日图 (按效应)', 'value': 'sunburst_effect'},
                    {'label': '矩形树图 (按体积)', 'value': 'treemap_volume'},
                    {'label': '矩形树图 (按效应)', 'value': 'treemap_effect'},
                ],
                value='sunburst_effect'  # 默认值
            ),
            html.Button('重新生成系统', id='regenerate-button', n_clicks=0, style={'marginTop': '20px'}),
            html.Button('演化系统 (扰动参数)', id='evolve-button', n_clicks=0,
                        style={'marginTop': '10px', 'marginLeft': '10px'}),
            html.Div(id='time-step-display', children="时间步: 0", style={'marginTop': '10px'}),

        ], className="four columns", style={'borderRight': '1px solid #ddd', 'paddingRight': '15px'}),

        # --- 可视化列 ---
        html.Div([
            dcc.Graph(id='fractal-viz', style={'height': '80vh'}),
            dcc.Markdown(id='summary-stats')  # 使用 dcc.Markdown 修复
        ], className="eight columns")
    ], className="row"),

    dcc.Store(id='evolution-params-store', data={'time_step': 0, 'perturb_factor': 0.01})
])


@app.callback(
    [Output('fractal-viz', 'figure'),
     Output('summary-stats', 'children'),  # 输出到 dcc.Markdown
     Output('time-step-display', 'children'),
     Output('evolution-params-store', 'data'),
     Output('p-eff-base-slider', 'value'),
     Output('eff-conc-factor-slider', 'value'),
     Output('ineff-spread-factor-slider', 'value'),
     Output('w-eff-base-slider', 'value'),
     Output('eff-effect-conc-slider', 'value'),
     Output('ineff-effect-damp-slider', 'value'),
     Output('rand-factor-slider', 'value'),
     ],
    [Input('regenerate-button', 'n_clicks'),
     Input('evolve-button', 'n_clicks')],
    [State('max-depth-slider', 'value'),
     State('p-eff-base-slider', 'value'),
     State('eff-conc-factor-slider', 'value'),
     State('ineff-spread-factor-slider', 'value'),
     State('w-eff-base-slider', 'value'),
     State('eff-effect-conc-slider', 'value'),
     State('ineff-effect-damp-slider', 'value'),
     State('rand-factor-slider', 'value'),
     State('viz-type-dropdown', 'value'),
     State('evolution-params-store', 'data')
     ]
)
def update_visualization_and_evolve(
        regen_clicks, evolve_clicks,
        max_depth, p_eff_base_val, eff_conc_factor_val, ineff_spread_factor_val,
        w_eff_base_val, eff_effect_conc_val, ineff_effect_damp_val, rand_factor_val,
        viz_type, evolution_data):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'

    p_eff_base = p_eff_base_val / 100.0
    w_eff_base = w_eff_base_val / 100.0
    eff_conc_factor = eff_conc_factor_val
    ineff_spread_factor = ineff_spread_factor_val
    eff_effect_conc = eff_effect_conc_val
    ineff_effect_damp = ineff_effect_damp_val
    rand_factor = rand_factor_val / 100.0

    current_time_step = evolution_data.get('time_step', 0)
    perturb_factor = evolution_data.get('perturb_factor', 0.01)

    if triggered_id == 'evolve-button':
        current_time_step += 1
        p_eff_base_val = max(5, min(50, p_eff_base_val + random.uniform(-1, 1) * 2))
        eff_conc_factor_val = max(0, min(10, eff_conc_factor_val + random.uniform(-0.5, 0.5)))
        w_eff_base_val = max(50, min(95, w_eff_base_val + random.uniform(-1, 1) * 2))

        p_eff_base = p_eff_base_val / 100.0
        w_eff_base = w_eff_base_val / 100.0
        eff_conc_factor = eff_conc_factor_val

    nodes = generate_fractal_data(
        max_depth, p_eff_base, eff_conc_factor, ineff_spread_factor,
        w_eff_base, eff_effect_conc, ineff_effect_damp, rand_factor, current_time_step
    )

    color_map_types = {"root": "lightgrey", "effective": "rgba(76, 175, 80, 0.7)",
                       "ineffective": "rgba(255, 152, 0, 0.7)"}
    node_colors_mapped = [color_map_types.get(type_val, "blue") for type_val in nodes["types"]]

    # 悬停文本 (Hover text)
    custom_data_hover = [
        (f"类型: {'有效' if t == 'effective' else ('无效' if t == 'ineffective' else '根')}<br>"
         f"深度: {d}<br>"
         f"体积: {v:.3E}<br>"
         f"效应: {e:.3E}<br>"
         f"效应密度: {ed:.2f}")
        for t, d, v, e, ed in
        zip(nodes["types"], nodes["depths"], nodes["volumes"], nodes["effects"], nodes["effect_densities"])
    ]

    fig = go.Figure()
    values_to_plot = []
    title_suffix_cn = ""

    if viz_type == 'sunburst_volume' or viz_type == 'treemap_volume':
        values_to_plot = nodes["volumes"]
        title_suffix_cn = "按体积"
    elif viz_type == 'sunburst_effect' or viz_type == 'treemap_effect':
        values_to_plot = nodes["effects"]
        title_suffix_cn = "按效应"

    chart_type_cn = "旭日图" if 'sunburst' in viz_type else "矩形树图"

    if 'sunburst' in viz_type:
        fig.add_trace(go.Sunburst(
            ids=nodes["ids"],
            labels=nodes["labels"],
            parents=nodes["parents"],
            values=values_to_plot,
            branchvalues="total",
            hovertext=custom_data_hover,
            hoverinfo="text+label+percent parent+percent root",  # Plotly 会自动处理这些标签的翻译（部分）
            marker_colors=node_colors_mapped,
            insidetextorientation='radial'
        ))
    elif 'treemap' in viz_type:
        fig.add_trace(go.Treemap(
            ids=nodes["ids"],
            labels=nodes["labels"],
            parents=nodes["parents"],
            values=values_to_plot,
            branchvalues="total",
            hovertext=custom_data_hover,
            hoverinfo="text+label+percent parent+percent root",
            marker_colors=node_colors_mapped,
        ))

    fig.update_layout(title_text=f"多重分形系统: {chart_type_cn} {title_suffix_cn}",
                      margin=dict(t=50, l=10, r=10, b=10))

    # 摘要统计
    leaf_volumes = sum(nodes["volumes"][i] for i, node_id in enumerate(nodes["ids"]) if
                       node_id not in nodes["parents"] and nodes["types"][i] != "root")
    leaf_effects = sum(nodes["effects"][i] for i, node_id in enumerate(nodes["ids"]) if
                       node_id not in nodes["parents"] and nodes["types"][i] != "root")
    num_nodes = len(nodes["ids"])

    # 使用 \n 进行换行，dcc.Markdown 会处理
    summary_text_md = f"""
总节点数: {num_nodes}
叶节点总体积: {leaf_volumes:.4f} (初始: {INITIAL_VOLUME})
叶节点总效应: {leaf_effects:.4f} (初始: {INITIAL_EFFECT})
"""

    updated_evolution_data = {'time_step': current_time_step, 'perturb_factor': perturb_factor}
    time_step_display_text = f"时间步: {current_time_step}"

    return (fig,
            summary_text_md,  # 直接传递 Markdown 字符串
            time_step_display_text,
            updated_evolution_data,
            p_eff_base_val, eff_conc_factor_val, ineff_spread_factor_val,
            w_eff_base_val, eff_effect_conc_val, ineff_effect_damp_val,
            rand_factor_val * 100
            )


if __name__ == '__main__':
    app.run(debug=True)