import dash
from dash import dcc, html, State, Dash, ctx as dash_ctx
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import random
import uuid
import math
import numpy as np
import time
import json

# --- 应用配置常量 ---
APP_TITLE = "三元结构多重分形系统浏览器 (RX-3.3)"  # 版本号更新
INITIAL_SYSTEM_VOLUME = 1.0
DEFAULT_MAX_DEPTH = 3
MAX_ALLOWED_DEPTH = 5
FACTOR_SENSITIVITY = 0.01  # 用于深度影响因子
MIN_PLOT_VALUE = 1e-9

# --- 参数默认值与范围 (三元结构) ---
PARAM_CONFIG_TRINARY = {
    "p_A_base": (10, 70, 1, 30, "A类基础体积比例 (%)", "A类（创造）节点的基础体积占比。"),
    "p_B_base": (10, 70, 1, 40, "B类基础体积比例 (%)", "B类（结构）节点的基础体积占比。"),
    # C类体积由 100 - A - B 间接确定

    "density_A_effect_base": (0.1, 5.0, 0.1, 2.0, "A类基础效应密度", "A类节点的基础效应密度。"),
    "depth_density_A_factor": (-1.0, 1.0, 0.1, -0.2, "A类密度深度影响因子",
                               f"每层深度，A类效应密度的变化量(乘以{FACTOR_SENSITIVITY})。负值表示随深度密度降低。"),

    "density_B_cost": (0.0, 2.0, 0.1, 0.5, "B类成本密度", "B类节点的维持成本密度。"),
    "density_C_cost": (0.0, 3.0, 0.1, 1.0, "C类成本密度", "C类节点的消耗成本密度。"),

    "depth_vol_factor_A": (
        -5.0, 5.0, 0.5, -1.0, "A类体积深度影响因子", f"每层深度，A类体积比例的变化量(乘以{FACTOR_SENSITIVITY})。"),
    "depth_vol_factor_B": (
        -5.0, 5.0, 0.5, 0.5, "B类体积深度影响因子", f"每层深度，B类体积比例的变化量(乘以{FACTOR_SENSITIVITY})。"),

    "rand_factor": (0.0, 10.0, 0.5, 1.0, "随机性因子 (%)", "为比例和密度计算增加随机扰动的最大幅度（百分比）。"),
    "evolve_perturb_factor": (0.1, 5.0, 0.1, 0.5, "演化扰动幅度 (%)", "点击演化时参数随机变化的最大百分比（相对范围）。")
}


# --- 辅助函数 ---
def get_trinary_proportions(current_depth, params):  # 比例计算
    # params 已经是计算用的小数值 (除了因子类本身)
    p_A_eff = params["p_A_base"] + current_depth * params["depth_vol_factor_A"] * FACTOR_SENSITIVITY
    p_B_eff = params["p_B_base"] + current_depth * params["depth_vol_factor_B"] * FACTOR_SENSITIVITY

    # 随机扰动应用于调整后的基础值
    rand_A_abs_effect = params["rand_factor"] * p_A_eff * 0.1  # 假设随机因子影响当前值的10%
    rand_B_abs_effect = params["rand_factor"] * p_B_eff * 0.1

    p_A = max(0.01, p_A_eff + random.uniform(-rand_A_abs_effect, rand_A_abs_effect))
    p_B = max(0.01, p_B_eff + random.uniform(-rand_B_abs_effect, rand_B_abs_effect))

    p_C_intermediate = 1.0 - p_A - p_B
    if p_C_intermediate < 0.01:
        scale_factor = (1.0 - 0.01) / (p_A + p_B) if (p_A + p_B) > 1e-6 else 1  # 避免除零
        p_A *= scale_factor;
        p_B *= scale_factor;
        p_C = 0.01
    else:
        p_C = p_C_intermediate

    total_p = p_A + p_B + p_C
    if total_p <= 1e-6: return 1 / 3, 1 / 3, 1 / 3  # 避免除零或极小值
    return p_A / total_p, p_B / total_p, p_C / total_p


def get_dynamic_density_A(current_depth, params):  # A类密度动态调整
    density = params["density_A_effect_base"] + current_depth * params["depth_density_A_factor"] * FACTOR_SENSITIVITY
    rand_effect = params["rand_factor"] * density * 0.05  # 假设随机因子影响当前值的5%
    return max(0.01, density + random.uniform(-rand_effect, rand_effect))  # 确保密度不为负或零


# --- 数据生成核心函数 ---
def generate_trinary_fractal_data(max_depth, generation_params_from_gui, current_time_step=0):
    nodes_data = {"ids": [], "labels": [], "parents": [], "volumes": [],
                  "effects_A": [], "costs_B": [], "costs_C": [], "net_effects": [],
                  "vis_values_net_effect": [], "types": [], "depths": []}
    node_counter = 0
    calc_params = {}
    for key, gui_val in generation_params_from_gui.items():  # GUI值转计算值
        if key in ["p_A_base", "p_B_base", "rand_factor"]:
            calc_params[key] = float(gui_val) / 100.0
        else:
            calc_params[key] = float(gui_val)  # 密度和因子直接用值
    start_time = time.time()

    def _recursive_generator(parent_id, parent_volume, current_depth):
        nonlocal node_counter
        if current_depth > max_depth: return 0, 0, 0, 0

        prop_A, prop_B, prop_C = get_trinary_proportions(current_depth, calc_params)

        # 获取当前深度的A类动态密度
        current_density_A = get_dynamic_density_A(current_depth, calc_params)
        # B和C的密度暂时保持固定 (从calc_params获取)
        current_density_B = calc_params["density_B_cost"]
        current_density_C = calc_params["density_C_cost"]

        node_types_props_densities = [
            ("A", prop_A, current_density_A, "创造"),
            ("B", prop_B, current_density_B, "结构"),
            ("C", prop_C, current_density_C, "消耗")]

        current_node_sum_A, current_node_sum_B, current_node_sum_C, current_node_sum_vis = 0, 0, 0, 0
        temp_children_data = []

        for type_code, proportion, density_val, type_label_prefix in node_types_props_densities:
            node_counter += 1
            child_id = f"节点_{type_code}{current_depth}_{node_counter}_{uuid.uuid4().hex[:4]}"
            child_volume = parent_volume * proportion

            gc_sum_A, gc_sum_B, gc_sum_C, gc_sum_vis = _recursive_generator(child_id, child_volume, current_depth + 1)

            dir_A, dir_B, dir_C = 0, 0, 0
            if current_depth == max_depth:  # 叶节点直接贡献
                if type_code == "A":
                    dir_A = child_volume * density_val
                elif type_code == "B":
                    dir_B = child_volume * density_val
                elif type_code == "C":
                    dir_C = child_volume * density_val

            child_total_A = dir_A + gc_sum_A;
            child_total_B = dir_B + gc_sum_B;
            child_total_C = dir_C + gc_sum_C
            child_net = child_total_A - child_total_B - child_total_C
            child_vis = gc_sum_vis if current_depth < max_depth else max(abs(child_net), MIN_PLOT_VALUE)
            if child_vis == 0: child_vis = MIN_PLOT_VALUE

            temp_children_data.append({
                "id": child_id, "label": f"L{current_depth}-{type_label_prefix} (V:{proportion * 100:.0f}%)",
                "parent": parent_id, "volume": child_volume, "effect_A": child_total_A, "cost_B": child_total_B,
                "cost_C": child_total_C, "net_effect": child_net, "vis_value": child_vis,
                "type": f"type_{type_code.lower()}", "depth": current_depth})

            current_node_sum_A += child_total_A;
            current_node_sum_B += child_total_B
            current_node_sum_C += child_total_C;
            current_node_sum_vis += child_vis

        for child_data in temp_children_data:  # 添加到全局列表
            for key_list, val_to_add in [
                (nodes_data["ids"], child_data["id"]), (nodes_data["labels"], child_data["label"]),
                (nodes_data["parents"], child_data["parent"]), (nodes_data["volumes"], child_data["volume"]),
                (nodes_data["effects_A"], child_data["effect_A"]), (nodes_data["costs_B"], child_data["cost_B"]),
                (nodes_data["costs_C"], child_data["cost_C"]), (nodes_data["net_effects"], child_data["net_effect"]),
                (nodes_data["vis_values_net_effect"], child_data["vis_value"]),
                (nodes_data["types"], child_data["type"]), (nodes_data["depths"], child_data["depth"])
            ]: key_list.append(val_to_add)

        return current_node_sum_A, current_node_sum_B, current_node_sum_C, current_node_sum_vis

    # 根节点处理
    total_A, total_B, total_C, root_vis_value = _recursive_generator("系统根节点", INITIAL_SYSTEM_VOLUME, 1)
    root_net_effect = total_A - total_B - total_C
    if root_vis_value == 0: root_vis_value = MIN_PLOT_VALUE

    # 插入根节点信息 (使用字典，然后各项插入，避免索引混乱)
    root_node_entry = {
        "ids": "系统根节点", "labels": f"系统根节点 (TS: {current_time_step})", "parents": "",
        "volumes": INITIAL_SYSTEM_VOLUME, "effects_A": total_A, "costs_B": total_B, "costs_C": total_C,
        "net_effects": root_net_effect, "vis_values_net_effect": root_vis_value,
        "types": "root", "depths": 0
    }
    for key, val in root_node_entry.items():
        nodes_data[key].insert(0, val)

    generation_time = time.time() - start_time
    return nodes_data, generation_time


# --- Dash 应用设置 ---
app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server;
app.title = APP_TITLE


# --- 构建GUI控件 ---
def generate_slider_marks_trinary(p_min, p_max, p_step):  # (与之前版本相同)
    if p_min == p_max: return {p_min: str(p_min)}
    marks = {p_min: str(p_min), p_max: str(p_max)}
    num_marks_approx = 5
    if num_marks_approx - 2 > 0:
        potential_marks = np.linspace(p_min, p_max, num_marks_approx)
        for val in potential_marks:
            marked_val = round(round(val / p_step) * p_step, 2) if p_step > 0 else round(val, 2)
            is_too_close = any(
                abs(marked_val - em_val) < p_step * 1.5 and marked_val != em_val for em_val in marks.keys())
            if not is_too_close: marks[marked_val] = str(marked_val) if marked_val % 1 != 0 else str(int(marked_val))
    return {float(k): v for k, v in marks.items()}


def build_trinary_param_controls():  # (与之前版本相同)
    controls = []
    for param_id, (p_min, p_max, p_step, p_val, p_label, p_desc) in PARAM_CONFIG_TRINARY.items():
        controls.append(html.Div([
            html.Label(p_label, style={'fontWeight': 'bold'}),
            dcc.Slider(id=f'{param_id}-slider', min=p_min, max=p_max, step=p_step, value=p_val,
                       marks=generate_slider_marks_trinary(p_min, p_max, p_step),
                       tooltip={"placement": "bottom", "always_visible": True}),
            dcc.Markdown(f"> _{p_desc}_", style={'fontSize': '0.85em', 'color': '#555', 'marginTop': '-5px'})
        ], style={'marginBottom': '18px'}))
    controls.append(dcc.Markdown("> _**C类体积** 由 (100% - A% - B%) 确定并归一化_",
                                 style={'fontSize': '0.8em', 'color': 'darkblue', 'border': '1px dashed blue',
                                        'padding': '5px'}))
    return controls


# --- 应用布局 ---
app.layout = html.Div([  # (与之前版本相同)
    html.H1(APP_TITLE, style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.P(f"提示: 最大深度超过 {MAX_ALLOWED_DEPTH - 1} (节点数 3^深度) 时性能可能急剧下降。",
           style={'textAlign': 'center', 'fontSize': '0.9em', 'color': 'darkred'}),
    html.Div([html.Div([html.H4("系统参数 (三元)",
                                style={'marginTop': '0px', 'borderBottom': '1px solid #eee', 'paddingBottom': '10px'}),
                        html.Label("最大深度:", style={'fontWeight': 'bold'}),
                        dcc.Slider(id='max-depth-slider', min=1, max=MAX_ALLOWED_DEPTH, step=1, value=DEFAULT_MAX_DEPTH,
                                   marks={i: str(i) for i in range(1, MAX_ALLOWED_DEPTH + 1)},
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        dcc.Markdown("> _分形递归的最大层数_",
                                     style={'fontSize': '0.85em', 'color': '#555', 'marginTop': '-5px'}),
                        html.Hr(style={'margin': '15px 0'}), *build_trinary_param_controls(),
                        html.Hr(style={'margin': '15px 0'}),
                        html.Label("可视化类型:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(id='viz-type-dropdown', options=[{'label': L, 'value': V} for L, V in
                                                                      [('旭日图(体积)', 'sunburst_volume'),
                                                                       ('旭日图(净效应)', 'sunburst_net_effect'),
                                                                       ('矩形树图(体积)', 'treemap_volume'),
                                                                       ('矩形树图(净效应)', 'treemap_net_effect')]],
                                     value='sunburst_net_effect', clearable=False),
                        html.Div(
                            [html.Button('重新生成', id='regenerate-button', n_clicks=0, style={'marginRight': '5px'}),
                             html.Button('演化系统', id='evolve-button', n_clicks=0, style={'marginRight': '5px'}),
                             html.Button('重置演化', id='reset-evolve-button', n_clicks=0)],
                            style={'marginTop': '20px', 'display': 'flex', 'justifyContent': 'space-around'}),
                        html.Div(id='time-step-display', children="时间步: 0",
                                 style={'marginTop': '12px', 'textAlign': 'center', 'fontWeight': 'bold'}),
                        ], className="four columns",
                       style={'borderRight': '1px solid #ddd', 'padding': '15px', 'maxHeight': '90vh',
                              'overflowY': 'auto', 'backgroundColor': '#f9f9f9'}),
              html.Div([dcc.Graph(id='fractal-viz', style={'height': 'calc(65vh - 30px)'}),
                        html.Div([dcc.Markdown(id='summary-stats', style={'whiteSpace': 'pre-wrap', 'padding': '8px',
                                                                          'borderTop': '1px solid #eee',
                                                                          'fontSize': '0.85em', 'maxHeight': '16vh',
                                                                          'overflowY': 'auto'}),
                                  html.Div(id='node-details-output', children=[html.P("点击图表中的节点查看详情。")],
                                           style={'padding': '8px', 'borderTop': '1px solid #ccc', 'marginTop': '8px',
                                                  'backgroundColor': '#f0f0f0', 'maxHeight': 'calc(19vh - 16px)',
                                                  'overflowY': 'auto', 'fontSize': '0.8em'})
                                  ], style={'height': 'calc(35vh)'})
                        ], className="eight columns", style={'padding': '15px'})
              ], className="row"),
    dcc.Store(id='evolution-state-store', data={'time_step': 0, 'last_params': None, 'generation_time_store': 0}),
    dcc.Store(id='generated-nodes-store', data=None)
])


# --- 回调函数辅助：解析参数 ---
def parse_trinary_callback_args(args):  # (与之前版本相同)
    idx = 0;
    regen_clicks = args[idx];
    idx += 1;
    evolve_clicks = args[idx];
    idx += 1;
    reset_evolve_clicks = args[idx];
    idx += 1
    graph_click_data = args[idx];
    idx += 1;
    max_depth = args[idx];
    idx += 1
    param_values_from_gui = {key: args[idx + i] for i, key in enumerate(PARAM_CONFIG_TRINARY.keys())};
    idx += len(PARAM_CONFIG_TRINARY.keys())
    viz_type = args[idx];
    idx += 1;
    evolution_state = args[idx];
    idx += 1;
    generated_nodes_json = args[idx];
    idx += 1
    return regen_clicks, evolve_clicks, reset_evolve_clicks, graph_click_data, max_depth, param_values_from_gui, viz_type, evolution_state, generated_nodes_json


# --- 主回调函数 ---
callback_inputs_trinary = [Input('regenerate-button', 'n_clicks'), Input('evolve-button', 'n_clicks'),
                           Input('reset-evolve-button', 'n_clicks'), Input('fractal-viz', 'clickData')]
callback_states_trinary = [State('max-depth-slider', 'value')]
for param_id in PARAM_CONFIG_TRINARY.keys(): callback_states_trinary.append(State(f'{param_id}-slider', 'value'))
callback_states_trinary.extend([State('viz-type-dropdown', 'value'), State('evolution-state-store', 'data'),
                                State('generated-nodes-store', 'data')])
callback_outputs_trinary = [Output('fractal-viz', 'figure'), Output('summary-stats', 'children'),
                            Output('time-step-display', 'children'), Output('evolution-state-store', 'data'),
                            Output('node-details-output', 'children'), Output('generated-nodes-store', 'data')]
for param_id in PARAM_CONFIG_TRINARY.keys(): callback_outputs_trinary.append(Output(f'{param_id}-slider', 'value'))


@app.callback(callback_outputs_trinary, callback_inputs_trinary, callback_states_trinary)
def update_trinary_app_state(*args):  # (回调逻辑与之前版本类似，但使用了新的参数和数据生成)
    (regen_clicks, evolve_clicks, reset_evolve_clicks, graph_click_data, max_depth,
     current_gui_params, viz_type, evolution_state, stored_nodes_json) = parse_trinary_callback_args(args)

    triggered_input_id = dash_ctx.triggered_id
    current_time_step = evolution_state.get('time_step', 0)
    current_generation_time = evolution_state.get('generation_time_store', 0)  # 使用存储的生成时间
    params_to_use = {k: float(v) for k, v in current_gui_params.items()}
    nodes = None;
    node_details_md = [html.P("点击图表中的节点查看详情。")]
    nodes_json_to_store = stored_nodes_json
    type_map = {"type_a": "A类(创造)", "type_b": "B类(结构)", "type_c": "C类(消耗)", "root": "根"}  # 修正type_map

    if triggered_input_id in ['regenerate-button', 'evolve-button',
                              'reset-evolve-button'] or dash_ctx.triggered_id is None:
        if triggered_input_id == 'evolve-button':  # 演化逻辑
            current_time_step += 1;
            perturb_factor = params_to_use["evolve_perturb_factor"] / 100.0
            for pid, (pmin, pmax, pstep, _, _, _) in PARAM_CONFIG_TRINARY.items():
                if pid == "evolve_perturb_factor": continue
                curr_val = params_to_use[pid];
                new_val = curr_val + random.uniform(-(pmax - pmin) * perturb_factor, (pmax - pmin) * perturb_factor) / 2
                if pstep > 0: new_val = round(new_val / pstep) * pstep
                params_to_use[pid] = max(pmin, min(pmax, new_val))
                # 精确处理浮点数小数位数，确保与step匹配
                if isinstance(pstep, float):
                    num_decimals = 0
                    if '.' in str(pstep): num_decimals = len(str(pstep).split('.')[1])
                    params_to_use[pid] = round(params_to_use[pid], num_decimals)

        elif triggered_input_id == 'reset-evolve-button':  # 重置逻辑
            current_time_step = 0
            for pid, (_, _, _, pval, _, _) in PARAM_CONFIG_TRINARY.items(): params_to_use[pid] = pval
        nodes, current_generation_time = generate_trinary_fractal_data(max_depth, params_to_use, current_time_step)
        nodes_json_to_store = json.dumps(nodes)
    elif triggered_input_id == 'fractal-viz' and stored_nodes_json:
        try:
            nodes = json.loads(stored_nodes_json)
        except:
            nodes = None
    if nodes is None:  # 确保在任何情况下都有nodes
        nodes, current_generation_time = generate_trinary_fractal_data(max_depth, params_to_use, current_time_step)
        nodes_json_to_store = json.dumps(nodes)

    if triggered_input_id == 'fractal-viz' and graph_click_data and nodes:  # 节点详情逻辑
        clicked_node_id = graph_click_data['points'][0].get('id')
        if clicked_node_id:
            try:
                node_idx = nodes["ids"].index(clicked_node_id)
                node_info_md = [html.H5(f"节点: {nodes['labels'][node_idx]}", style={'marginBottom': '5px'})]
                details = {"ID": nodes["ids"][node_idx], "类型": type_map.get(nodes["types"][node_idx], "未知"),
                           "深度": nodes["depths"][node_idx], "体积": f"{nodes['volumes'][node_idx]:.2E}",
                           "A效(子树)": f"{nodes['effects_A'][node_idx]:.2E}",
                           "B成本(子树)": f"{nodes['costs_B'][node_idx]:.2E}",
                           "C成本(子树)": f"{nodes['costs_C'][node_idx]:.2E}",
                           "净效(子树)": f"{nodes['net_effects'][node_idx]:.2E}"}
                for k, v_item in details.items():  # 使用 v_item 避免与外层作用域可能的 v 冲突
                    node_info_md.append(html.P(f"{k}: {v_item}", style={'margin': '1px 0'}))
                children_info = [
                    f"{nodes['labels'][i].split('(')[0]}(V:{nodes['volumes'][i]:.1E}, NE:{nodes['net_effects'][i]:.1E})"
                    for i, pid in enumerate(nodes["parents"]) if pid == clicked_node_id]
                if children_info:
                    node_info_md.append(html.H6("子节点:", style={'marginTop': '6px', 'marginBottom': '2px'}))
                    for cs in children_info: node_info_md.append(
                        html.P(cs, style={'margin': '1px 0', 'fontSize': '0.9em'}))
                else:
                    node_info_md.append(html.P("无子节点", style={'marginTop': '6px', 'fontStyle': 'italic'}))
                node_details_md = node_info_md
            except ValueError:
                node_details_md = [html.P(f"错误: 未找到ID {clicked_node_id}", style={'color': 'red'})]
        else:
            node_details_md = [html.P("点击数据中未找到节点ID", style={'color': 'orange'})]

    fig = go.Figure()  # 图表生成逻辑
    if nodes:
        color_map_trinary = {"root": "#888888", "type_a": "#2ca02c", "type_b": "#1f77b4", "type_c": "#d62728"}
        node_colors_mapped = [color_map_trinary.get(type_val, "#ff7f0e") for type_val in nodes["types"]]
        custom_data_hover_trinary = [(f"类型:{type_map.get(t, '未知')}<br>深度:{d}<br>体积:{v:.2E}<br>"
                                      f"A效:{ea:.2E},B成本:{cb:.2E},C成本:{cc:.2E}<br>净效:{ne:.2E}")
                                     for t, d, v, ea, cb, cc, ne in
                                     zip(nodes["types"], nodes["depths"], nodes["volumes"],
                                         nodes["effects_A"], nodes["costs_B"], nodes["costs_C"], nodes["net_effects"])]
        values_to_plot = nodes["volumes"] if 'volume' in viz_type else nodes["vis_values_net_effect"]
        title_suffix_cn = "按体积" if 'volume' in viz_type else "按净效应(可视化值)"
        chart_type_cn = "旭日图" if 'sunburst' in viz_type else "矩形树图"
        common_params = dict(ids=nodes["ids"], labels=nodes["labels"], parents=nodes["parents"], values=values_to_plot,
                             branchvalues="total", hovertext=custom_data_hover_trinary,
                             hoverinfo="text+label+percent parent", marker_colors=node_colors_mapped)
        if 'sunburst' in viz_type:
            fig.add_trace(go.Sunburst(**common_params, insidetextorientation='radial'))
        elif 'treemap' in viz_type:
            fig.add_trace(go.Treemap(**common_params))
        fig.update_layout(title_text=f"{APP_TITLE}: {chart_type_cn} {title_suffix_cn}",
                          margin=dict(t=50, l=10, r=10, b=10))
    else:
        fig.update_layout(title_text="请生成系统", xaxis={"visible": False}, yaxis={"visible": False})

    summary_text_md = f"**系统摘要 (TS: {current_time_step})**\n---\n"  # 摘要统计
    if nodes:
        num_nodes_val = len(nodes["ids"]);
        root_vol, root_eff_A, root_cost_B, root_cost_C, root_net_eff = (
            nodes["volumes"][0], nodes["effects_A"][0], nodes["costs_B"][0], nodes["costs_C"][0],
            nodes["net_effects"][0])
        summary_text_md += f"总节点数: {num_nodes_val} (生成: {current_generation_time:.3f}s)\n"
        summary_text_md += f"根体积: {root_vol:.3f}\n根净效应: {root_net_eff:.3f} (A:{root_eff_A:.2f}-B:{root_cost_B:.2f}-C:{root_cost_C:.2f})"
    else:
        summary_text_md += "无数据。"

    updated_evolution_state = {'time_step': current_time_step, 'last_params': params_to_use,
                               'generation_time_store': current_generation_time}
    time_step_display_text = f"时间步: {current_time_step}"
    returned_slider_values = []  # 滑块值返回
    for key in list(PARAM_CONFIG_TRINARY.keys()):
        val = params_to_use[key]
        p_min, p_max, p_step, _, _, _ = PARAM_CONFIG_TRINARY[key]
        if isinstance(p_step, float):
            val = float(val)
            if p_step > 0:  # 只有当step > 0 时才尝试根据小数位数round
                num_decimals = 0
                if '.' in str(p_step): num_decimals = len(str(p_step).split('.')[1])
                if num_decimals > 0:
                    val = round(val, num_decimals)
                else:
                    val = round(val)  # 如果step是像1.0这样的，则round到整数
            # else: step is 0 or negative, keep val as is (or handle error)
        else:
            val = int(round(float(val)))
        returned_slider_values.append(val)

    return tuple([fig, summary_text_md, time_step_display_text, updated_evolution_state,
                  node_details_md, nodes_json_to_store] + returned_slider_values)


if __name__ == '__main__':
    app.run(debug=True, port=8052)
