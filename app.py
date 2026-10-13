import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import io
import json
import re
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pyvis.network import Network
import pickle

# حل مشكلة دعم الحروف العربية والـ Unicode
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==========================================
# 1. إعدادات واجهة Streamlit
# ==========================================
st.set_page_config(page_title="Dr. Skin - AI Dermatologist", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
    body, .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Tajawal', sans-serif !important;
        background-color: #F3F6F8;
    }
    .main-header {
        color: #1A365D;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: -20px;
        margin-bottom: 30px;
    }
    .stChatMessage {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        margin-bottom: 15px;
        padding: 10px 15px;
    }
    div[data-testid="chatAvatarIcon-user"] { background-color: #3182CE; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-left: 1px solid #E2E8F0; }
    h3 { color: #2D3748; font-weight: 700; border-bottom: 2px solid #EDF2F7; padding-bottom: 10px; margin-bottom: 20px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. كلاسات الذكاء الاصطناعي (KG & GAT)
# ==========================================
class SkincareKnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def call_knowledge_graph(self, skin_type, active_problems, category_filter=None):
        if category_filter is None:
            category_filter = ["Cleanser", "Toner", "Serum", "Treatment", "Moisturizer", "Essence", "Mask", "Sunscreen"]
        avoid_ingredients, recommended_ingredients = set(), set()
        skin_en = re.sub(r'\(.*?\)', '', skin_type).strip()

        if skin_type in self.G:
            for neighbor in self.G.neighbors(skin_type):
                if self.G[skin_type][neighbor]['relation'] == 'AVOIDS': avoid_ingredients.add(neighbor)
        if skin_en in self.G and skin_en != skin_type:
            for neighbor in self.G.neighbors(skin_en):
                if self.G[skin_en][neighbor]['relation'] == 'AVOIDS': avoid_ingredients.add(neighbor)
        for prob in active_problems:
            if prob in self.G:
                for neighbor in self.G.neighbors(prob):
                    if self.G[prob][neighbor]['relation'] == 'AVOIDS': avoid_ingredients.add(neighbor)

        for prob in active_problems:
            for node in self.G.nodes():
                if self.G.has_edge(node, prob):
                    rel = self.G[node][prob]['relation']
                    if rel in ['TREATS_BY_RULE', 'TREATS_BY_INGREDIENT'] and node not in avoid_ingredients:
                        recommended_ingredients.add(node)

        safe_products = []
        for product in self.G.nodes():
            if self.G.nodes[product].get('type') == 'Product':
                prod_category = self.G.nodes[product].get('category', '')
                if not any(cat.lower() in prod_category.lower() for cat in category_filter): continue
                is_suitable, contains_active, contains_avoided = False, False, False
                for n in self.G.neighbors(product):
                    if self.G[product][n]['relation'] == 'SUITABLE_FOR' and (skin_en.lower() in n.lower() or 'all types' in n.lower()):
                        is_suitable = True
                        break
                if is_suitable:
                    for n in self.G.neighbors(product):
                        if self.G[product][n]['relation'] == 'CONTAINS':
                            if any(banned.lower() in n.lower() for banned in avoid_ingredients): contains_avoided = True
                            if any(rec.lower() in n.lower() for rec in recommended_ingredients): contains_active = True
                    if contains_active and not contains_avoided: safe_products.append(product)

        return {"Skin_Type_Detected": skin_type, "Problems_Detected": active_problems, "Safe_Ingredients": list(recommended_ingredients), "Recommended_Products": safe_products[:5]}

class SkincareGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=64, heads=8, dropout=0.3):
        super(SkincareGAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x

    def get_embeddings(self, graph_data):
        self.eval()
        with torch.no_grad(): return self.forward(graph_data.x, graph_data.edge_index)

def compute_multihop_score(product_name, problem_name, skin_type, embeddings, node_to_id, kg_tool, alpha=0.4):
    G = kg_tool.G
    prob_emb, product_emb = embeddings[node_to_id[problem_name]], embeddings[node_to_id[product_name]]
    direct_score = F.cosine_similarity(prob_emb.unsqueeze(0), product_emb.unsqueeze(0)).item()
    bridge_scores = [F.cosine_similarity(prob_emb.unsqueeze(0), embeddings[node_to_id[neighbor]].unsqueeze(0)).item() for neighbor in G.neighbors(product_name) if G[product_name][neighbor].get('relation', '') == 'CONTAINS' and neighbor in node_to_id]
    bridge_score = max(bridge_scores) if bridge_scores else 0.0
    return {"final_score": round(alpha * direct_score + (1 - alpha) * bridge_score, 4), "direct_score": round(direct_score, 4), "bridge_score": round(bridge_score, 4)}

def gnn_recommend_pro(problem_name, skin_type, graph_data, gat_model, node_to_id, id_to_node, kg_tool, top_k=5, diverse=True):
    safe_products = set(kg_tool.call_knowledge_graph(skin_type=skin_type, active_problems=[problem_name])["Recommended_Products"])
    if not safe_products or problem_name not in node_to_id: return []

    embeddings = gat_model.get_embeddings(graph_data)
    scored_products = [{"product": p, "category": kg_tool.G.nodes[p].get('category', 'N/A'), **compute_multihop_score(p, problem_name, skin_type, embeddings, node_to_id, kg_tool)} for p in safe_products if p in node_to_id]
    scored_products.sort(key=lambda x: x["final_score"], reverse=True)

    if diverse:
        seen, diverse_list, leftover = set(), [], []
        for p in scored_products: (diverse_list if p["category"] not in seen and not seen.add(p["category"]) else leftover).append(p)
        return (diverse_list + leftover)[:top_k]
    return scored_products[:top_k]

# ==========================================
# 3. وكلاء LangGraph
# ==========================================
class SkincareState(TypedDict):
    user_message: str
    chat_history: str
    skin_type: str
    active_problems: List[str]
    raw_recommendations: List[dict]
    final_response: str

def triage_agent(state: SkincareState):
    current_skin, current_prob = state.get('skin_type', 'Unknown'), state.get('active_problems', ['Unknown'])[0]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "أنت طبيب ذكي. المعلومات: البشرة {current_skin} والمشكلة {current_prob}. سياق المحادثة: {chat_history}. استخرج المشكلة (من: Acne (General), Dehydrated Skin, Hyperpigmentation, Wrinkles, Rosacea) ونوع البشرة (من: Sensitive, Dry, Oily, Combination, Normal). أعد JSON فقط: {{\"skin_type\": \"...\", \"problem\": \"...\"}}"),
        ("user", "{message}")
    ])
    response = (prompt | llm).invoke({"message": state["user_message"], "current_skin": current_skin, "current_prob": current_prob, "chat_history": state.get('chat_history', '')})
    text = response.content[0].get("text", "") if isinstance(response.content, list) else str(response.content)
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    ext = json.loads(match.group(0)) if match else {"skin_type": current_skin, "problem": current_prob}
    return {"skin_type": ext.get('skin_type', current_skin), "active_problems": [ext.get('problem', current_prob)]}

def engine_agent(state: SkincareState):
    results = gnn_recommend_pro(state['active_problems'][0], state['skin_type'], graph_data_g, gat_model_g, node_to_id_g, id_to_node_g, kg_tool_g, top_k=3)
    return {"raw_recommendations": results}

def pharmacist_agent(state: SkincareState):
    skin, prob = state.get('skin_type', 'Unknown'), state.get('active_problems', ['Unknown'])[0]
    instruction = "البيانات مكتملة. اشرح المنتجات بأسلوب طبي مهني." if skin != "Unknown" and prob != "Unknown" else "ينقصك إما تحديد نوع البشرة أو المشكلة. تفاعل مع المريض واسأله بلطف لتحديدها."
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"أنت مستشار عناية بالبشرة طبي ومهني (دكتور سكين). البشرة: {{skin}}، المشكلة: {{prob}}.\nالتعليمات: {instruction}\nسياق الحديث:\n{{chat_history}}\nقواعد: كن رسمياً ولطيفاً، لا تستخدم إيموجي مبالغ فيها، واهتم بصحة المريض."),
        ("user", "{user_message}")
    ])
    response = (prompt | llm).invoke({"skin": skin, "prob": prob, "chat_history": state.get('chat_history', ''), "user_message": state.get('user_message', '')})
    return {"final_response": response.content[0].get("text", "") if isinstance(response.content, list) else str(response.content)}

def route_after_triage(state: SkincareState):
    return "Pharmacist" if state.get('skin_type', 'Unknown') == "Unknown" or state.get('active_problems', ['Unknown'])[0] == "Unknown" else "Engine"

# ==========================================
# 4. محرك الخريطة التفاعلية
# ==========================================
def draw_interactive_graph(state, kg_tool):
    net = Network(height="620px", width="100%", bgcolor="#FFFFFF", font_color="#2D3748")
    skin = state.get('skin_type', 'Unknown')
    prob = state.get('active_problems', ['Unknown'])[0]
    recs = state.get('raw_recommendations', [])

    active_nodes = {}
    active_edges = set()
    active_nodes["مريضي"] = {'color': "#1A365D", 'size': 35, 'label': "أنت (المريض)"}

    if skin != 'Unknown':
        skin_en = re.sub(r'\(.*?\)', '', skin).strip()
        active_nodes[skin] = {'color': "#3182CE", 'size': 30, 'label': f"البشرة: {skin}"}
        active_nodes[skin_en] = {'color': "#3182CE", 'size': 30, 'label': f"البشرة: {skin_en}"}
        active_edges.add(("مريضي", skin))
        active_edges.add(("مريضي", skin_en))

    if prob != 'Unknown':
        active_nodes[prob] = {'color': "#E53E3E", 'size': 30, 'label': f"المشكلة: {prob}"}
        active_edges.add(("مريضي", prob))

    if recs:
        for r in recs:
            prod_name = r['product']
            active_nodes[prod_name] = {'color': "#D69E2E", 'size': 40, 'label': f"منتج: {prod_name}"}
            active_edges.add(("مريضي", prod_name))
            if skin != 'Unknown':
                active_edges.add((prod_name, skin))
                active_edges.add((prod_name, skin_en))

            ingredients_added = 0
            for neighbor in kg_tool.G.neighbors(prod_name):
                if kg_tool.G[prod_name][neighbor].get('relation', '') == 'CONTAINS':
                    if prob != 'Unknown' and kg_tool.G.has_edge(neighbor, prob):
                        active_nodes[neighbor] = {'color': "#38A169", 'size': 25, 'label': f"مكون: {neighbor}"}
                        active_edges.add((prod_name, neighbor))
                        active_edges.add((neighbor, prob))
                        ingredients_added += 1
                    if ingredients_added >= 3: break

    for node, attrs in kg_tool.G.nodes(data=True):
        if node in active_nodes:
            props = active_nodes[node]
            net.add_node(node, label=props['label'], title=str(node), color=props['color'], size=props['size'], font={'color': '#1A202C', 'size': 14, 'bold': True})
        else:
            net.add_node(node, label=" ", title=f"{attrs.get('type', 'عنصر')}: {node}", color="#EDF2F7", size=12)

    net.add_node("مريضي", label="أنت (المريض)", title="المريض الحالي", color="#1A365D", size=35, font={'color': '#1A202C', 'size': 16, 'bold': True})

    for source, target, data in kg_tool.G.edges(data=True):
        if (source, target) in active_edges or (target, source) in active_edges:
            net.add_edge(source, target, color="#4A5568", width=3)
        else:
            net.add_edge(source, target, color="#E2E8F0", width=0.5)

    if skin != 'Unknown' and skin in active_nodes: net.add_edge("مريضي", skin, color="#3182CE", width=3)
    if prob != 'Unknown' and prob in active_nodes: net.add_edge("مريضي", prob, color="#E53E3E", width=3)
    if recs:
        for r in recs: net.add_edge("مريضي", r['product'], color="#D69E2E", width=2)

    net.set_options('{"physics": {"solver": "forceAtlas2Based", "forceAtlas2Based": {"gravitationalConstant": -60, "centralGravity": 0.015, "springLength": 100, "springConstant": 0.08, "damping": 0.8}, "minVelocity": 0.75, "stabilization": {"enabled": true, "iterations": 100, "fit": true}}}')
    net.save_graph("kg_visualization.html")
    with open("kg_visualization.html", 'r', encoding='utf-8') as f: return f.read()

# ==========================================
# 5. تهيئة النظام والواجهة
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_system(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    llm_instance = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, api_key=api_key)

    kg_instance = SkincareKnowledgeGraph()
    with open('kg_graph.pkl', 'rb') as f: kg_instance.G = pickle.load(f)

    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        n2i, i2n = mappings['node_to_id'], mappings['id_to_node']

    g_data = torch.load('graph_data.pt', weights_only=False)
    g_model = SkincareGAT(in_channels=g_data.num_node_features, hidden_channels=64, out_channels=64, heads=8, dropout=0.3)
    g_model.load_state_dict(torch.load('gat_model_weights.pth', weights_only=True))
    g_model.eval()

    workflow = StateGraph(SkincareState)
    workflow.add_node("Triage", triage_agent)
    workflow.add_node("Engine", engine_agent)
    workflow.add_node("Pharmacist", pharmacist_agent)
    workflow.set_entry_point("Triage")
    workflow.add_conditional_edges("Triage", route_after_triage, {"Engine": "Engine", "Pharmacist": "Pharmacist"})
    workflow.add_edge("Engine", "Pharmacist")
    workflow.add_edge("Pharmacist", END)

    return {
        "app": workflow.compile(checkpointer=MemorySaver()), "llm": llm_instance, "kg": kg_instance,
        "g_data": g_data, "g_model": g_model, "n2i": n2i, "i2n": i2n
    }

# ========================================================
# 🔑 مفتاح الـ API:
# -
# ========================================================
MY_API_KEY = "AQ.Ab8RN6L1i9PhrrMpH327IOrnNuXoHGeWsSx9ZZNncf6Gg04_Bw"

st.markdown("<h1 class='main-header'>🩺 الطبيب الذكي للعناية بالبشرة</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=120)
    st.markdown("<h2 style='text-align: center; color: #1A365D;'>عن العيادة الذكية</h2>", unsafe_allow_html=True)
    st.info("هذا النظام يعتمد على الذكاء الاصطناعي الرمزي (Neuro-Symbolic AI) لتقديم استشارات طبية موثوقة وآمنة لعلاج مشاكل البشرة.")
    st.markdown("---")
    st.markdown("**فريق التطوير:**")
    st.markdown("👩‍💻 لونا الشيخ")
    st.markdown("👨‍💻 علي صقور")

with st.spinner("⏳ جاري تحضير العيادة..."):
    sys_resources = initialize_system(MY_API_KEY)
    chatbot_app = sys_resources["app"]
    global llm, kg_tool_g, graph_data_g, gat_model_g, node_to_id_g, id_to_node_g
    llm, kg_tool_g, graph_data_g = sys_resources["llm"], sys_resources["kg"], sys_resources["g_data"]
    gat_model_g, node_to_id_g, id_to_node_g = sys_resources["g_model"], sys_resources["n2i"], sys_resources["i2n"]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك في العيادة الذكية 🩺. كيف يمكنني مساعدتك اليوم؟ يرجى إخباري بمشكلة بشرتك ونوعها لكي أصف لك العلاج المناسب.", "state_snapshot": {}}]

col_chat, col_graph = st.columns([1.1, 1], gap="large")

with col_chat:
    st.markdown("### 💬 الاستشارة الطبية")
    chat_container = st.container(height=520, border=False)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("state_snapshot") and msg["state_snapshot"].get("raw_recommendations"):
                    with st.expander("🔬 تحليل التوصية الطبي (GAT)"):
                        for r in msg["state_snapshot"]["raw_recommendations"]:
                            st.caption(f"💊 **{r['product']}** (الدقة: {r['final_score']:.2f})")

    if prompt := st.chat_input("اكتب رسالتك للطبيب هنا..."):
        st.session_state.messages.append({"role": "user", "content": prompt, "state_snapshot": {}})

        with chat_container:
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("الطبيب يقوم بتحليل حالتك... 🧠"):
                    history_str = "".join([f"{'المريض' if m['role']=='user' else 'الطبيب'}: {m['content']}\n" for m in st.session_state.messages[-6:]])

                    final_state = chatbot_app.invoke({
                        "user_message": prompt,
                        "chat_history": history_str,
                        "skin_type": st.session_state.get('current_skin', 'Unknown'),
                        "active_problems": st.session_state.get('current_prob', ['Unknown'])
                    }, config={"configurable": {"thread_id": "st_001"}})

                    st.write(final_state['final_response'])

                    st.session_state['current_skin'] = final_state.get('skin_type', 'Unknown')
                    st.session_state['current_prob'] = final_state.get('active_problems', ['Unknown'])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_state['final_response'],
                        "state_snapshot": final_state
                    })
        st.rerun()

with col_graph:
    st.markdown("### 🧬 الخريطة المعرفية للعلاج")
    st.caption("رسم بياني تفاعلي يوضح كيفية تفكير النظام وربط المشكلة بالمكونات والعلاج.")

    latest_state = st.session_state.messages[-1].get("state_snapshot", {})
    if not latest_state:
        latest_state = {
            'skin_type': st.session_state.get('current_skin', 'Unknown'),
            'active_problems': st.session_state.get('current_prob', ['Unknown']),
            'raw_recommendations': []
        }

    html_graph = draw_interactive_graph(latest_state, kg_tool_g)
    st.markdown("""<div style="background-color: white; padding: 10px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E2E8F0;">""", unsafe_allow_html=True)
    components.html(html_graph, height=500)
    st.markdown("</div>", unsafe_allow_html=True)
