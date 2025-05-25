import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, Set
# from sklearn.metrics.pairwise import cosine_similarity # Not used in current version
import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import json
import re
from dotenv import load_dotenv
import time

load_dotenv()

# API config, take kimi for example
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
MOONSHOT_API_URL = "https://api.moonshot.cn/v1/chat/completions"

# --- Font setup for Matplotlib (if SimHei is available) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese characters
    plt.rcParams['axes.unicode_minus'] = False  # Properly display negative signs
except Exception as e:
    print(f"SimHei font not found, Chinese characters in plots might not display correctly: {e}")


class TCMKnowledgeGraph:
    def __init__(self, csv_path: str = None):
        self.graph = nx.MultiDiGraph()
        self.triples_with_source = []
        self.relation_types = set()
        self.entity_types = {} # Placeholder for future entity typing
        self.treatment_plans_details = {} # Pre-aggregated details for quick lookup

        if csv_path:
            self.load_from_csv(csv_path)

    def load_from_csv(self, csv_path: str):
        try:
            df = pd.read_csv(csv_path)
            print(f"成功加载CSV文件，共有{len(df)}条三元组数据")

            required_columns = ['Subject', 'Predicate', 'Object']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"CSV文件缺少必要的列：{col}")

            for _, row in tqdm(df.iterrows(), total=len(df), desc="构建知识图谱"):
                subject = str(row['Subject']).strip()
                predicate = str(row['Predicate']).strip()
                obj = str(row['Object']).strip()

                source_book = str(row.get('SourceBookName', '')).strip()
                source_chapter = str(row.get('SourceChapterName', '')).strip()

                if source_book.lower() == 'nan': source_book = ''
                if source_chapter.lower() == 'nan': source_chapter = ''
                
                source_info_str = "未知来源"
                book_display = ""
                chap_display = ""

                if source_book:
                    book_display = f"《{source_book}》"
                if source_chapter:
                    chap_display = source_chapter
                
                if book_display and chap_display:
                    source_info_str = f"{book_display} - {chap_display}"
                elif book_display:
                    source_info_str = book_display
                elif chap_display:
                    source_info_str = chap_display

                self.add_triple(subject, predicate, obj, source_book, source_chapter, source_info_str)

            print(f"成功构建知识图谱，共有{len(self.graph.nodes)}个节点，{len(self.graph.edges)}条边")
            print(f"关系类型：{self.relation_types}")

        except Exception as e:
            print(f"加载CSV文件失败：{e}")
            raise

    def add_triple(self, subject: str, predicate: str, obj: str,
                   source_book: str = "", source_chapter: str = "", # Default to empty if truly unknown
                   source_info_str: str = "未知来源"):
        if not subject or not predicate or not obj: # Basic validation
            # print(f"Skipping triple due to empty field: S='{subject}', P='{predicate}', O='{obj}'")
            return

        self.graph.add_node(subject) #, type='unknown') # Future: add type
        self.graph.add_node(obj) #, type='unknown')

        self.graph.add_edge(subject, obj, relation=predicate, source=source_info_str)
        self.relation_types.add(predicate)
        self.triples_with_source.append((subject, predicate, obj, source_book, source_chapter, source_info_str))

        # Pre-aggregation for known treatment plan structures
        # Updated relation names based on the first script's output
        treatment_related_predicates = [
            "使用药材", "剂量", "制备方法", "备注", "表现症状", "描述疾病", # Added "描述疾病"
            "定义治疗方案", "治疗疾病", "治疗症状" # "定义治疗方案" instead of "描述治疗方案"
        ]
        # Check if subject is a treatment plan (heuristic)
        if ("治疗方案" in subject or "方剂" in subject or "汤" in subject or "散" in subject or "丸" in subject) and \
           predicate in treatment_related_predicates:
            if subject not in self.treatment_plans_details:
                self.treatment_plans_details[subject] = {}
            if predicate not in self.treatment_plans_details[subject]:
                self.treatment_plans_details[subject][predicate] = []
            
            # Avoid duplicate objects for the same predicate under a plan
            is_duplicate = any(item["value"] == obj for item in self.treatment_plans_details[subject][predicate])
            if not is_duplicate:
                 self.treatment_plans_details[subject][predicate].append({"value": obj, "source": source_info_str})

    def get_entity_relations(self, entity: str, specific_relations: Optional[List[str]] = None) -> List[Tuple[str, str, str, str]]:
        relations = []
        for s, p, o, _, _, source_str in self.triples_with_source: # Iterate full list for comprehensive source
            if s == entity or o == entity:
                if specific_relations is None or p in specific_relations:
                    relations.append((s, p, o, source_str))
        return relations

    def get_treatment_plan_full_details(self, plan_name: str) -> Dict[str, Any]:
        details = {
            "名称": plan_name, "组成": [], "制备方法": [], "功能主治": [],
            "备注": [], "相关症状": [], "治疗疾病": [], "来源信息": set()
        }
        # Relation mapping from the first script's 'target_relations_map' keys
        # "使用药材", "治疗疾病", "治疗症状", "制备方法", "备注" (Note: "制备方法" not in original list, "备注" is also new here but common)
        # "表现症状" is more for Disease -> Symptom. For Plan -> Symptom, "治疗症状" is more direct.

        # 1. From preprocessed dictionary
        if plan_name in self.treatment_plans_details:
            for pred, obj_source_list in self.treatment_plans_details[plan_name].items():
                for item in obj_source_list:
                    obj_val, source_str = item["value"], item["source"]
                    if source_str != "未知来源": details["来源信息"].add(source_str)

                    if pred == "使用药材":
                        if not any(d.get("药材") == obj_val for d in details["组成"]):
                             details["组成"].append({"药材": obj_val, "剂量": "未知", "来源": source_str})
                    elif pred == "制备方法": details["制备方法"].append({"value": obj_val, "source": source_str})
                    elif pred == "备注": details["备注"].append({"value": obj_val, "source": source_str})
                    elif pred == "治疗疾病": details["功能主治"].append({"value": obj_val, "source": source_str}) # Also for "功能主治"
                    elif pred == "治疗症状": details["相关症状"].append({"value": obj_val, "source": source_str})


        # 2. Graph traversal (1-hop) for potentially missed or differently structured info
        if plan_name in self.graph:
            for _, target, edge_data in self.graph.out_edges(plan_name, data=True):
                relation = edge_data['relation']
                source_str = edge_data.get('source', "未知来源")
                if source_str != "未知来源": details["来源信息"].add(source_str)

                if relation == "使用药材":
                    if not any(d["药材"] == target for d in details["组成"]): # Avoid duplicates from step 1
                        details["组成"].append({"药材": target, "剂量": "未知", "来源": source_str})
                elif relation == "制备方法" and not any(d["value"] == target for d in details["制备方法"]):
                    details["制备方法"].append({"value": target, "source": source_str})
                elif relation == "备注" and not any(d["value"] == target for d in details["备注"]):
                    details["备注"].append({"value": target, "source": source_str})
                elif relation == "治疗疾病" and not any(d["value"] == target for d in details["功能主治"]):
                    details["功能主治"].append({"value": target, "source": source_str})
                elif relation == "治疗症状" and not any(d["value"] == target for d in details["相关症状"]):
                    details["相关症状"].append({"value": target, "source": source_str})
        
        # 3. Get dosage for herbs (another hop)
        updated_composition = []
        for herb_item in details["组成"]:
            herb_name = herb_item["药材"]
            herb_dosages_with_source = []
            # Search for (herb_name)-[剂量]->(dosage_value)
            if herb_name in self.graph:
                for _, dosage_val_node, edge_data_dosage in self.graph.out_edges(herb_name, data=True):
                    if edge_data_dosage.get("relation") == "剂量":
                        dosage_source = edge_data_dosage.get("source", "未知来源")
                        herb_dosages_with_source.append({"value": dosage_val_node, "source": dosage_source})
                        if dosage_source != "未知来源": details["来源信息"].add(dosage_source)
            
            if herb_dosages_with_source:
                herb_item["剂量"] = "; ".join(ds["value"] for ds in herb_dosages_with_source)
                herb_item["剂量来源"] = sorted(list(set(ds["source"] for ds in herb_dosages_with_source if ds["source"] != "未知来源")))
            updated_composition.append(herb_item)
        details["组成"] = updated_composition
        
        details["来源信息"] = sorted(list(details["来源信息"]))
        details = {k: v for k, v in details.items() if v or k == "名称"} # Keep name even if other fields empty
        return details

    def search_by_keyword(self, keyword: str) -> List[str]:
        matched_entities = []
        if not keyword: return []
        keyword_l = keyword.lower()
        for entity in self.graph.nodes():
            entity_str = str(entity)
            if keyword_l == entity_str.lower() or keyword_l in entity_str.lower():
                matched_entities.append(entity_str)
        return list(set(matched_entities))

    def get_related_entities(self, entity: str, relation_type: Optional[str] = None, max_depth: int = 1) -> List[str]:
        if entity not in self.graph.nodes():
            return []
        
        visited = {entity}
        # queue stores (node, depth)
        queue = [(entity, 0)]
        related_entities = []
        
        head = 0
        while head < len(queue):
            current_entity, depth = queue[head]
            head += 1

            if depth >= max_depth:
                continue

            # Outgoing edges
            for _, neighbor, edge_data in self.graph.out_edges(current_entity, data=True):
                if relation_type and edge_data.get('relation') != relation_type:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    related_entities.append(neighbor)
            
            # Incoming edges
            for source, _, edge_data in self.graph.in_edges(current_entity, data=True):
                if relation_type and edge_data.get('relation') != relation_type:
                    continue
                if source not in visited:
                    visited.add(source)
                    queue.append((source, depth + 1))
                    related_entities.append(source)
                    
        return list(set(related_entities))

    # --- NEW METHOD for semantic path finding ---
    def find_semantic_paths(self,
                            start_node: str,
                            end_node_keywords: List[str],
                            max_hops: int = 3,
                            # intermediate_relation_preference: Optional[List[str]] = None, # Could be added later
                            avoid_cycles_in_path_nodes: bool = True
                           ) -> List[Dict[str, Any]]:
        if start_node not in self.graph:
            return []

        found_paths_details = []
        # queue stores: (current_node_name, list_of_path_segments_so_far)
        # path_segment: {"s":_,"p":_,"o":_,"src":_}
        queue = [(start_node, [])] 
        
        # To avoid re-processing (node, path_prefix_tuple) pairs if graph is very loopy
        # For simpler cycle avoidance in path, we check node names in current_path_nodes
        # visited_bfs_states = set() 

        head = 0
        while head < len(queue):
            current_node_name, current_path_segments = queue[head]
            head += 1

            current_path_nodes = [start_node] + [seg["o"] for seg in current_path_segments]
            current_hop_count = len(current_path_segments)

            # Check if current_node_name is a target
            is_target_node = False
            if end_node_keywords: # If specific targets are given
                for keyword in end_node_keywords:
                    if keyword.lower() in current_node_name.lower():
                        is_target_node = True
                        break
            elif not end_node_keywords and current_hop_count > 0 : # If no specific target, any node at >0 hops can be an end
                is_target_node = True


            if is_target_node and current_path_segments: # Path must have at least one segment
                path_nodes_list = [start_node] + [seg["o"] for seg in current_path_segments]
                relations_list = [seg["p"] for seg in current_path_segments]
                sources_set = set(seg["src"] for seg in current_path_segments if seg["src"] and seg["src"] != "未知来源")
                
                path_str_parts = []
                for i, seg in enumerate(current_path_segments):
                    s_node = path_nodes_list[i]
                    p_rel = seg["p"]
                    o_node = seg["o"]
                    src_info = seg["src"] if seg["src"] and seg["src"] != "未知来源" else "未知"
                    path_str_parts.append(f"('{s_node}')-[{p_rel} (源: {src_info})]->")
                path_visual = "".join(path_str_parts[:-1]) + f"('{path_nodes_list[-1]}')" # Fix last segment display

                if not path_str_parts: # Handle case for path A to A (0 hops, no segments)
                     path_visual = f"('{start_node}')"


                found_paths_details.append({
                    "path_nodes": path_nodes_list,
                    "relations": relations_list,
                    "path_visual": path_visual,
                    "sources": sorted(list(sources_set)),
                    "start_node": start_node,
                    "end_node": current_node_name,
                    "hops": current_hop_count
                })
            
            if current_hop_count >= max_hops:
                continue

            # Explore neighbors (outgoing edges)
            if current_node_name in self.graph:
                for _, neighbor, edge_data in self.graph.out_edges(current_node_name, data=True):
                    if avoid_cycles_in_path_nodes and neighbor in current_path_nodes:
                        continue
                    
                    new_segment = {
                        "s": current_node_name,
                        "p": edge_data.get('relation', '未知关系'),
                        "o": neighbor,
                        "src": edge_data.get('source', '未知来源')
                    }
                    new_path_segments = current_path_segments + [new_segment]
                    queue.append((neighbor, new_path_segments))
        
        # Deduplicate paths based on visual representation and sort
        # (This BFS might find multiple paths to the same node via different routes, which is good)
        unique_paths_final = []
        seen_visuals = set()
        for p_detail in sorted(found_paths_details, key=lambda x: (x['hops'], -len(x['sources']))):
            if p_detail['path_visual'] not in seen_visuals:
                unique_paths_final.append(p_detail)
                seen_visuals.add(p_detail['path_visual'])
        return unique_paths_final

    def visualize_graph(self, entities: List[str] = None, figsize: Tuple[int, int] = (18, 15)): # Increased figsize
        sub_graph_nodes = set()
        if entities:
            for entity in entities:
                if entity in self.graph:
                    sub_graph_nodes.add(entity)
                    # Add immediate neighbors for context
                    for _, neighbor in self.graph.out_edges(entity):
                        sub_graph_nodes.add(neighbor)
                    for predecessor, _ in self.graph.in_edges(entity):
                        sub_graph_nodes.add(predecessor)
            g = self.graph.subgraph(list(sub_graph_nodes)).copy()
        else: # Visualize a sample if no entities given
            if len(self.graph.nodes()) > 50:
                nodes = list(self.graph.nodes())
                # Prioritize nodes with higher degree for sampling if possible
                degrees = dict(self.graph.degree())
                sorted_nodes_by_degree = sorted(nodes, key=lambda n: degrees.get(n, 0), reverse=True)
                sampled_nodes = sorted_nodes_by_degree[:50]
                
                # Expand sample to include neighbors for better connectivity
                expanded_sample = set(sampled_nodes)
                for snode in sampled_nodes:
                    for _, neigh in self.graph.out_edges(snode): expanded_sample.add(neigh)
                    for pred, _ in self.graph.in_edges(snode): expanded_sample.add(pred)
                g = self.graph.subgraph(list(expanded_sample)).copy()

            else:
                g = self.graph.copy()

        # Remove isolated nodes from the subgraph to focus on relationships
        g.remove_nodes_from(list(nx.isolates(g)))

        if not g.nodes():
            print("子图为空或仅包含孤立节点，无法可视化。")
            return

        plt.figure(figsize=figsize)
        try:
            # Kamada-Kawai can be slow for larger subgraphs
            if len(g.nodes()) < 100 :
                 pos = nx.kamada_kawai_layout(g)
            else:
                 pos = nx.spring_layout(g, k=0.25, iterations=30, seed=42) # Adjusted for larger graphs
        except Exception as e_layout:
            print(f"Layout algorithm failed ({e_layout}), using spring_layout as fallback.")
            pos = nx.spring_layout(g, k=0.25, iterations=30, seed=42)

        nx.draw_networkx_nodes(g, pos, node_size=350, alpha=0.8, node_color='skyblue',_linewidths=0.5)
        nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.4, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=12, connectionstyle='arc3,rad=0.05')
        nx.draw_networkx_labels(g, pos, font_size=9) # Font family set globally

        edge_labels_dict = {}
        for u, v, data in g.edges(data=True):
            # For MultiDiGraph, multiple edges can exist between u,v. We take one.
            # To show all, one might need to adjust label positions or aggregate.
            if (u,v) not in edge_labels_dict: # Show one label per pair for clarity
                 edge_labels_dict[(u,v)] = data.get('relation', '')
            # else: # If wanting to show multiple relations for the same (u,v) pair
            #    edge_labels_dict[(u,v)] += f"\n{data.get('relation', '')}"


        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels_dict, font_size=7, alpha=0.9, bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.5))


        plt.title("中医知识子图可视化", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class GraphRAG:
    def __init__(self, knowledge_graph: TCMKnowledgeGraph):
        self.kg = knowledge_graph
        self.llm_cache = {}

    def query_moonshot_api(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        cache_key = f"{prompt}_{temperature}_{max_tokens}"
        if cache_key in self.llm_cache:
            # print("LLM从缓存加载")
            return self.llm_cache[cache_key]

        if not MOONSHOT_API_KEY or "sk-your" in MOONSHOT_API_KEY: # Basic check
             print("错误: MOONSHOT_API_KEY 未配置或无效。")
             return "API Key未配置或无效。"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MOONSHOT_API_KEY}"
        }
        data = {
            "model": "moonshot-v1-32k", # Or choose other models like moonshot-v1-8k, moonshot-v1-128k
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(MOONSHOT_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            result = response.json()
            
            if "choices" not in result or not result["choices"]:
                print(f"API响应格式错误或无choice: {result}")
                return "API响应错误，未找到回答。"
            
            response_text = result["choices"][0]["message"]["content"]
            self.llm_cache[cache_key] = response_text
            return response_text
        except requests.exceptions.Timeout:
            print("调用API超时。")
            return "调用API超时，请稍后再试或检查网络连接。"
        except requests.exceptions.HTTPError as e:
            print(f"API HTTP错误: {e} - {response.text}")
            return f"API请求失败: {e}"
        except Exception as e:
            print(f"调用API失败: {e}")
            return f"调用API时发生错误: {str(e)}"

    def extract_keywords_and_intent(self, query: str) -> Dict[str, Any]:
        prompt = f"""
        请分析以下中医相关查询，提取核心实体词（如中药名、症状、疾病名、方剂名等），并判断用户的主要意图。
        主要意图可以是："查询实体信息"、"寻求治疗方案"、"比较实体"、"解释概念"、"未知"。
        请按JSON格式输出，包含 "keywords" (字符串列表) 和 "intent" (字符串) 两个字段。

        查询: "{query}"

        JSON输出:
        """
        response_text = self.query_moonshot_api(prompt, temperature=0.1, max_tokens=512)
        try:
            # Handle potential markdown code blocks
            clean_response = re.sub(r"```json\s*|\s*```", "", response_text, flags=re.MULTILINE).strip()
            result = json.loads(clean_response)
            if not isinstance(result.get("keywords"), list):
                result["keywords"] = []
            if not isinstance(result.get("intent"), str):
                result["intent"] = "未知"
            return result
        except json.JSONDecodeError:
            print(f"关键词和意图提取JSON解析失败: {response_text[:500]}") # Log part of response
            # Fallback: attempt to parse keywords if response is just a list of words
            keywords = [kw.strip() for kw in response_text.splitlines() if kw.strip() and '{' not in kw and '}' not in kw and ':' not in kw]
            return {"keywords": keywords if keywords else [query], "intent": "未知"}


    # --- MODIFIED retrieve_relevant_knowledge ---
    def retrieve_relevant_knowledge(self, query: str, extracted_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        keywords = extracted_info.get("keywords", [])
        intent = extracted_info.get("intent", "未知")
        # print(f"提取的关键词: {keywords}, 用户意图: {intent}")

        relevant_knowledge_items: List[Dict[str, Any]] = []
        # Entities for which we've already retrieved detailed (treatment plan) or path info, to avoid redundancy
        processed_entities_for_context = set() 

        all_matched_initial_entities = []
        for keyword in keywords:
            matched = self.kg.search_by_keyword(keyword)
            all_matched_initial_entities.extend(matched)
        unique_start_entities = list(set(all_matched_initial_entities))
        # print(f"知识图谱中匹配到的初始实体: {unique_start_entities}")

        candidate_plans_for_details = []
        is_seeking_treatment_intent = (intent == "寻求治疗方案") or \
                                    any(treat_kw in query.lower() for treat_kw in ["怎么办", "如何治", "用什么药", "治疗方法", "方剂"])

        # 1. Direct treatment plan retrieval (if intent matches or keywords suggest it)
        if is_seeking_treatment_intent:
            for entity_name in unique_start_entities: # From keyword matches
                if "治疗方案" in entity_name or "方剂" in entity_name or \
                   entity_name.endswith("汤") or entity_name.endswith("散") or entity_name.endswith("丸"):
                    if entity_name not in processed_entities_for_context:
                        candidate_plans_for_details.append(entity_name)
                        # processed_entities_for_context.add(entity_name) # Add when details are actually fetched

            # 1-hop reverse/forward search for plans from symptom/disease keywords
            symptom_disease_keywords = [kw for kw in keywords if not ("治疗方案" in kw or "方剂" in kw or kw.endswith("汤") or kw.endswith("散") or kw.endswith("丸"))]
            for keyword_sd in symptom_disease_keywords:
                for s, p, o, _, _, _ in self.kg.triples_with_source:
                    is_plan_subject = "治疗方案" in s or "方剂" in s or s.endswith("汤") or s.endswith("散") or s.endswith("丸")
                    is_plan_object = "治疗方案" in o or "方剂" in o or o.endswith("汤") or o.endswith("散") or o.endswith("丸")

                    if o.lower() == keyword_sd.lower() and \
                       ("治疗疾病" in p or "治疗症状" in p or "主治" in p or "用于治疗" in p) and \
                       is_plan_subject and s not in candidate_plans_for_details and s not in processed_entities_for_context:
                        candidate_plans_for_details.append(s)
                    elif s.lower() == keyword_sd.lower() and \
                         ("推荐方剂" in p or "定义治疗方案" in p or "宜用" in p) and \
                         is_plan_object and o not in candidate_plans_for_details and o not in processed_entities_for_context:
                        candidate_plans_for_details.append(o)
            
            candidate_plans_for_details = list(set(candidate_plans_for_details)) # Deduplicate
            # print(f"初步候选治疗方案列表 (直接+1跳): {candidate_plans_for_details}")

            MAX_PLANS_TO_DETAIL = 5 # Limit detailed plans to keep context concise
            detailed_plans_count = 0
            for plan_name in candidate_plans_for_details:
                if detailed_plans_count >= MAX_PLANS_TO_DETAIL: break
                if plan_name not in processed_entities_for_context: # Check again
                    plan_details = self.kg.get_treatment_plan_full_details(plan_name)
                    if plan_details.get("组成") or plan_details.get("功能主治"): # Ensure plan has some substance
                        relevant_knowledge_items.append({"type": "治疗方案", "name": plan_name, "details": plan_details})
                        detailed_plans_count += 1
                        processed_entities_for_context.add(plan_name)
        
        # 2. Multi-hop path retrieval
        MAX_PATHS_TO_SHOW = 5 # Limit number of paths in context
        found_multi_hop_paths_details = []
        
        path_end_target_keywords = ["治疗方案", "方剂", "药材", "草药", "汤", "散", "丸"] # Default targets
        if "症状" in query.lower() or "疾病" in query.lower(): # If query is about symptoms/diseases, broaden targets
             path_end_target_keywords.extend(["证候", "病机"]) # Look for explanations

        if unique_start_entities: # Only if we have starting points from keywords
            # print(f"开始进行多跳路径检索，从 {unique_start_entities} 到含 {path_end_target_keywords} 的节点...")
            temp_paths_collected = []
            for start_entity in unique_start_entities:
                if start_entity in processed_entities_for_context : # Avoid re-processing if already detailed
                    continue
                # If start_entity itself is a plan/herb, don't find paths from it to the same type
                is_start_plan_herb = any(ptk.lower() in start_entity.lower() for ptk in ["治疗方案", "方剂", "药材", "草药","汤","散","丸"])
                
                current_paths = self.kg.find_semantic_paths(start_entity, 
                                                            path_end_target_keywords if not is_start_plan_herb else [], # if start is plan/herb, require different end types implicitly
                                                            max_hops=3)
                temp_paths_collected.extend(current_paths)
                if len(temp_paths_collected) > MAX_PATHS_TO_SHOW * 2: # Early exit if too many paths found
                    break
            
            # Sort and select best paths (e.g., shorter, more sources, more relevant end nodes)
            # Simple sort: hops, then number of sources (desc)
            # Deduplicate visually similar paths if find_semantic_paths didn't fully handle it
            unique_visual_paths = {}
            for p_detail in sorted(temp_paths_collected, key=lambda x: (x['hops'], -len(x.get('sources',[])))):
                if p_detail['path_visual'] not in unique_visual_paths:
                    # Further filter: ensure end_node is not the same as start_node for meaningful paths
                    if p_detail['start_node'] != p_detail['end_node']:
                         unique_visual_paths[p_detail['path_visual']] = p_detail
            
            found_multi_hop_paths_details = list(unique_visual_paths.values())[:MAX_PATHS_TO_SHOW]

        if found_multi_hop_paths_details:
            relevant_knowledge_items.append({
                "type": "相关推导路径",
                "paths": found_multi_hop_paths_details
            })
            for path_info in found_multi_hop_paths_details: # Mark nodes in paths as processed for context
                processed_entities_for_context.add(path_info["start_node"])
                processed_entities_for_context.add(path_info["end_node"])
            # print(f"找到 {len(found_multi_hop_paths_details)} 条多跳路径。")

        # 3. Basic 1-hop triples as supplementary info, if other contexts are sparse
        # Only add if we have few items, or if specific keywords didn't yield rich context above
        # And avoid showing direct relations for entities already covered by plans/paths
        needs_supplementary_triples = not relevant_knowledge_items or \
                                      len(relevant_knowledge_items) < 2 or \
                                      (len(relevant_knowledge_items) == 1 and relevant_knowledge_items[0]["type"] == "相关推导路径" and not relevant_knowledge_items[0].get("paths"))


        if needs_supplementary_triples:
            basic_triples_context = []
            MAX_TRIPLES_PER_ENTITY = 1
            TOTAL_MAX_BASIC_TRIPLES = 3
            triples_collected_this_round = 0

            for entity in unique_start_entities:
                if entity in processed_entities_for_context: continue # Skip if already part of detailed context
                if triples_collected_this_round >= TOTAL_MAX_BASIC_TRIPLES: break

                relations_for_entity = self.kg.get_entity_relations(entity) # (s,p,o,source_str)
                ranked_relations = self._rank_triples_by_relevance(relations_for_entity, query, keywords)
                
                added_for_this_entity = 0
                for s, p, o, src_str in ranked_relations:
                    # Avoid triples where subject and object are the same, or too trivial
                    if s == o: continue
                    if triples_collected_this_round >= TOTAL_MAX_BASIC_TRIPLES or added_for_this_entity >= MAX_TRIPLES_PER_ENTITY:
                        break
                    basic_triples_context.append((s,p,o,src_str))
                    triples_collected_this_round +=1
                    added_for_this_entity +=1
                processed_entities_for_context.add(entity) # Mark as processed for this basic triple step

            if basic_triples_context:
                if not any(item.get("type") == "三元组列表" for item in relevant_knowledge_items):
                     relevant_knowledge_items.append({"type": "三元组列表", "triples": basic_triples_context})
        
        # Prioritize knowledge items: Detailed Plans > Paths > Basic Triples
        # This can be done by ordering the append operations or by sorting `relevant_knowledge_items` later
        # For now, the append order somewhat reflects this.
        MAX_KNOWLEDGE_ITEMS_FINAL = 3 # Max items to pass to LLM context formatter
        return relevant_knowledge_items[:MAX_KNOWLEDGE_ITEMS_FINAL]


    def _rank_triples_by_relevance(self, triples_with_source: List[Tuple[str, str, str, str]],
                                   query: str, keywords: List[str]) -> List[Tuple[str, str, str, str]]:
        scored_triples = []
        for triple_item in triples_with_source: # triple_item is (s,p,o,source_str)
            s, p, o, _ = triple_item # Unpack only s,p,o for scoring
            score = 0
            # Higher score if keyword is subject or object
            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in s.lower(): score += 5
                if kw_l in o.lower(): score += 3
                if kw_l in p.lower(): score += 1 # Relations are also important
            
            # Bonus for words from query not in keywords (general relevance)
            for word in query.split():
                word_l = word.lower()
                if word_l not in [k.lower() for k in keywords]: # Avoid double counting
                    if word_l in s.lower(): score += 2
                    if word_l in o.lower(): score += 1
            
            # Prefer triples with known sources
            if triple_item[3] != "未知来源":
                score += 0.5 

            # Simple length penalty for overly long object strings (often definitions, less direct relation)
            if len(o) > 100 : score -= 0.5
            if len(s) > 100 : score -= 0.5


            scored_triples.append((triple_item, score))
        
        # Sort by score descending
        return [t[0] for t in sorted(scored_triples, key=lambda x: x[1], reverse=True)]

    # --- MODIFIED _format_knowledge_for_llm ---
    def _format_knowledge_for_llm(self, knowledge_items: List[Dict[str, Any]]) -> str:
        formatted_text = ""
        if not knowledge_items:
            return "未从知识库中检索到与查询直接相关的详细中医知识。"

        item_count = 0
        MAX_ITEMS_IN_CONTEXT = 3 # Hard cap on items being formatted for LLM

        for item in knowledge_items:
            if item_count >= MAX_ITEMS_IN_CONTEXT: break
            
            current_item_text = ""
            if item["type"] == "治疗方案":
                item_count += 1
                details = item["details"]
                current_item_text += f"【治疗方案 {item_count}: {details.get('名称', item.get('name','未知方案'))}】\n"

                plan_sources = details.get("来源信息", [])
                if plan_sources:
                    source_display = plan_sources[0] if len(plan_sources) == 1 else '; '.join(plan_sources[:3]) # Limit sources display
                    current_item_text += f"  主要来源: {source_display}\n"

                if details.get("功能主治"):
                    current_item_text += f"  功能主治:\n"
                    for val_src_item in details["功能主治"][:3]: # Limit items
                        source_tag = f"(来源: {val_src_item['source']})" if val_src_item.get('source') and val_src_item['source'] != "未知来源" else ""
                        current_item_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                if details.get("相关症状"):
                    current_item_text += f"  相关症状:\n"
                    for val_src_item in details["相关症状"][:3]:
                        source_tag = f"(来源: {val_src_item['source']})" if val_src_item.get('source') and val_src_item['source'] != "未知来源" else ""
                        current_item_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                if details.get("组成"):
                    current_item_text += "  组成:\n"
                    for comp in details["组成"][:5]: # Limit ingredients
                        dose_str = comp.get('剂量', '未知')
                        dose_source_str = ""
                        if comp.get("剂量来源"):
                            unique_dose_sources = sorted(list(set(comp['剂量来源'])))
                            if unique_dose_sources and unique_dose_sources != ["未知来源"]:
                                dose_source_str = f" (剂量来源: {'; '.join(unique_dose_sources[:2])})" # Limit dose sources
                        
                        herb_source_str = f"(药材条目来源: {comp['来源']})" if comp.get('来源') and comp['来源'] != "未知来源" else ""
                        current_item_text += f"    - {comp['药材']} (剂量: {dose_str}{dose_source_str}) {herb_source_str}\n".strip() + "\n"
                
                for detail_key, display_name in [("制备方法", "制备方法"), ("备注", "备注")]:
                    if details.get(detail_key):
                        current_item_text += f"  {display_name}:\n"
                        for val_src_item in details[detail_key][:2]: # Limit items
                            source_tag = f"(来源: {val_src_item['source']})" if val_src_item.get('source') and val_src_item['source'] != "未知来源" else ""
                            current_item_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                current_item_text += "\n"
            
            elif item["type"] == "相关推导路径" and item.get("paths"):
                item_count +=1
                current_item_text += f"【相关知识路径推导 {item_count}】:\n"
                path_idx = 0
                for path_detail in item["paths"][:MAX_PATHS_TO_SHOW]: # Already limited in retrieve, but good to double check
                    path_idx += 1
                    source_tag = f"(综合来源: {'; '.join(path_detail.get('sources',[]))})" if path_detail.get('sources') else ""
                    path_desc = f"路径 {path_idx} (从 '{path_detail.get('start_node','未知起点')}' 到 '{path_detail.get('end_node','未知终点')}', {path_detail.get('hops','未知')}跳): "
                    current_item_text += f"  {path_desc}{path_detail.get('path_visual','路径描述错误')} {source_tag}\n"
                current_item_text += "\n"

            elif item["type"] == "三元组列表" and item.get("triples"):
                item_count += 1
                current_item_text += f"【其他相关知识点 {item_count}】:\n"
                for s, p, o, source_str in item["triples"][:TOTAL_MAX_BASIC_TRIPLES]: # Already limited, but double check
                    source_tag = f"(来源: {source_str})" if source_str and source_str != "未知来源" else ""
                    current_item_text += f"  - “{s}” {p} “{o}” {source_tag}\n".strip() + "\n"
                current_item_text += "\n"
            
            if current_item_text: # Only add to formatted_text if something was generated for this item
                 formatted_text += current_item_text


        if not formatted_text.strip():
            return "未从知识库中检索到足够详细的中医知识来回答该问题。" # Fallback message
        return formatted_text.strip()

    def generate_graphrag_response_only(self, query: str, include_context_debug: bool = False) -> Tuple[str, str, List[Dict[str, Any]], Dict[str,Any]]:
        extracted_info = self.extract_keywords_and_intent(query)
        # print(f"DEBUG: Extracted Info: {extracted_info}") # Debug
        relevant_knowledge_items = self.retrieve_relevant_knowledge(query, extracted_info)
        # print(f"DEBUG: Relevant Knowledge Items: {json.dumps(relevant_knowledge_items, ensure_ascii=False, indent=2)}") # Debug
        context_for_llm = self._format_knowledge_for_llm(relevant_knowledge_items)
        # print(f"DEBUG: Context for LLM: {context_for_llm}") # Debug
        intent = extracted_info.get("intent", "未知")

        no_kg_context_messages = [
            "未从知识库中检索到与查询直接相关的详细中医知识。",
            "未从知识库中检索到足够详细的中医知识来回答该问题。"
        ]
        if context_for_llm.strip() in no_kg_context_messages or not relevant_knowledge_items:
            response_text = context_for_llm # Return the "not found" message
            # print("GraphRAG: No specific context found in KG or items list empty.")
            return response_text, context_for_llm, relevant_knowledge_items, extracted_info

        prompt_template = f"""
        你是一位严谨的中医药文献研究员。请【严格基于】以下提供的“中医知识库上下文”，清晰、准确地回答用户的问题。

        【中医知识库上下文】：
        {context_for_llm}
        ---
        用户问题：“{query}”
        ---
        请遵循以下指示：
        1.  你的回答【必须完全基于】上述“中医知识库上下文”。【不要补充上下文之外的任何信息】。
        2.  如果上下文中包含治疗方案的详细组成、剂量等，或相关的知识路径，请在回答中【明确列出这些细节】。
        3.  在你的回答中，如果引用了“中医知识库上下文”中的具体信息点，请在该信息点或句末【必须】使用括号注明其来源，格式为“(来源: 《书籍名》- 章节名)”或“(综合来源: ...路径来源...)”。
        4.  回答应尽可能【精炼且直接】。如果上下文信息足以回答，请直接给出答案；如果不足，请指出信息有限或未在上下文中找到。
        5.  注意排版，可以使用项目符号使回答更清晰。
        """
        is_seeking_treatment_intent = (intent == "寻求治疗方案") or \
                                   any(treat_kw in query.lower() for treat_kw in ["怎么办", "如何治", "用什么药", "治疗方法", "方剂"])
        if is_seeking_treatment_intent:
            prompt_template += """
        [关于治疗方案的额外指示]：如果上下文中提到了与用户问题相关的治疗方案，请详细说明其【名称、主要功能主治、组成药材（及剂量，若有）、制备方法/备注（若有）】，并务必标注各项信息的来源。如果提供了相关知识路径，也请清晰地阐述。
            """
        prompt_template += "\n请给出你的回答："
        
        # if include_context_debug: # For manual prompt inspection
        #      print(f"--- Prompt for GraphRAG Response Only (Temperature: 0.1) ---\n{prompt_template}\n-------------------------------------------------------------")

        response_text = self.query_moonshot_api(prompt_template, temperature=0.1, max_tokens=2000)

        if include_context_debug: # This will now print to console from the app
            debug_output = (
                f"---DEBUG: GraphRAG Context Generation---\n"
                f"意图：{intent}\n关键词：{extracted_info.get('keywords')}\n"
                f"检索到的知识项数量：{len(relevant_knowledge_items)}\n"
                # f"检索到的知识项：{json.dumps(relevant_knowledge_items, ensure_ascii=False, indent=2)}\n" # Can be very verbose
                f"---CONTEXT FOR GraphRAG-LLM---\n{context_for_llm}\n"
                # f"---GraphRAG-LLM RESPONSE---\n{response_text}" # Response already printed by caller
            )
            print(debug_output)
        
        return response_text, context_for_llm, relevant_knowledge_items, extracted_info

    def get_general_kimi_response(self, query: str, temperature: float = 0.5, max_tokens: int = 2048) -> str:
        prompt = f"""
        你是一位知识渊博且资深的中医专家。请针对以下用户提出的问题，提供一个全面、详细、且结构清晰的解答。
        请根据具体问题（药物/病症查询、功能/作用等）尽可能从不同方面（例如：定义、病因病机、主要类型、常见症状、诊断要点、治疗原则、常用方药举例、预后转归、生活调理及注意事项等，根据问题类型酌情选择）且有重点地进行阐述。
        请确保语言专业、严谨，同时也要易于理解。
        
        用户问题： “{query}”

        你的专业解答：
        """
        return self.query_moonshot_api(prompt, temperature=temperature, max_tokens=max_tokens)
    
    def synthesize_responses(self, query: str, graphrag_response: str, general_response: str,
                             temperature: float = 0.3, max_tokens: int = 3000) -> str:

        is_graphrag_valid = not any(
            msg in graphrag_response for msg in [
                "未从知识库中检索到与查询直接相关的详细中医知识。",
                "未从知识库中检索到足够详细的中医知识来回答该问题。",
                "API Key未配置或无效。", # Added API error check
                "API请求失败",
                "API响应错误"
            ]
        ) and graphrag_response.strip() and len(graphrag_response.strip()) > 10 # Basic length check

        synthesis_prompt = f"""
        你是一位资深的中医内容编辑和撰稿专家。你的任务是将以下两份关于用户问题“{query}”的回答，巧妙地融合成一个单一、全面、准确、行文流畅自然、且带有清晰来源标注的最终专业答案。

        【第一份回答】（此回答主要基于一个专门的古籍中医知识图谱检索，并由AI初步总结，其核心价值在于包含具体的古籍文献来源信息和细节）：
        ---
        {graphrag_response if is_graphrag_valid else "（本地知识图谱中未找到与问题直接相关的具体记载或检索时发生错误）"}
        ---

        【第二份回答】（此回答来自一个通用的大型语言模型，特点是知识面较广，解释可能更系统和详尽，但通常不包含具体的古籍文献来源）：
        ---
        {general_response}
        ---

        请遵循以下【严格的整合指示】进行操作：

        1.  **结构与内容主干**：以【第二份回答】（通用回答）的结构、广度和系统性论述作为整合后答案的基础框架和主要叙述流程。
        2.  **细节融入与深度增强**：
            * 将【第一份回答】（图谱版回答）中的【所有具体的、有价值的信息点，如药物组成、剂量、特定的治疗方法细节、知识路径推导、古籍中的独特论述，以及最重要的——文献来源标注】（例如“(来源: 《金匮要略》- 某某篇)”或“(综合来源: ...)”）——【准确无误且极为自然地】融入到【第二份回答】的相应论述段落或知识点中。
            * **核心目标**：让这些来自古籍或知识图谱的信息看起来像是对通用回答中相应论点的**原生补充、具体例证、或深化解释**，而不是生硬的插入或独立的附加信息块。力求使整合后的文本浑然一体。
        3.  **来源信息的处理与呈现**：
            * 如果两份回答中存在信息重叠（例如，对同一功效或方剂的描述），【优先采用并整合来自第一份回答中带有文献来源的表述和细节】，用其来丰富、具体化或替代通用回答中的对应内容。
            * 所有来自第一份回答的文献来源标注都必须保留，并清晰地附在相应的信息点之后。确保来源格式清晰。来源标注应作为信息点阐述完毕后的自然收尾，避免突兀感。
        4.  **处理图谱无特定信息的情况**：如果【第一份回答】明确指出“未找到信息”或内容为空/无效，则最终答案主要依赖【第二份回答】。此时，无需刻意提及知识图谱未找到信息，除非你认为这样的说明对用户有益。
        5.  **内容补充与文脉扩展**：
            * 如果【第一份回答】提供了【第二份回答】中完全没有提及的【与问题相关的具体治疗方案、药物组成、剂量、知识路径、古籍观点等重要细节】，请务必将这些有价值的内容【无缝地、合乎逻辑地整合】到最终答案的恰当部分。如果具体剂量和制备方法未给出则省略不显示。
            * 这可能需要你对【第二份回答】的局部结构进行微调或扩展，以确保新增信息的融入既自然又保持了整体论述的连贯性和流畅性。
        6.  **专业性、可读性与文风统一**：
            * 最终答案应保持中医的专业术语准确性，同时语言表达应流畅自然、结构清晰，易于普通用户理解（可适当使用项目符号、分点阐述等方式优化排版）。
            * **至关重要的是**：确保整合后的全文文风统一、语调一致，避免出现两种回答风格的明显割裂感，让读者感觉这是由一位专家一气呵成撰写的内容。同时，对于读者问题中的关注点（某种药物/病症的查询、功能、作用、定义等）应在突出相应的重点。
        7.  **标准提醒**：在最终答案的末尾，务必加上标准的用药提醒：“请注意，以上信息仅供参考，具体用药和治疗方案请务必咨询专业中医师进行辨证论治，切勿自行用药。”

        请基于以上所有指示，以专业的判断和高超的编辑技巧，输出整合后的【最终专业答案】：
        """
        # print(f"--- Synthesis Prompt ---\nGraphRAG valid: {is_graphrag_valid}\nGraphRAG Response for Synthesis:\n{graphrag_response if is_graphrag_valid else '图谱无效'}\n---END SYNTH PROMPT ---") # Debug
        return self.query_moonshot_api(synthesis_prompt, temperature=temperature, max_tokens=max_tokens)

class TCMGraphRAGApp:
    def __init__(self, csv_path: str):
        if not MOONSHOT_API_KEY or "sk-your" in MOONSHOT_API_KEY :
            print("警告: 未设置有效MOONSHOT_API_KEY环境变量，API调用将失败。")
        print("正在加载中医知识图谱...")
        self.kg = TCMKnowledgeGraph(csv_path)
        self.rag = GraphRAG(self.kg)
        print("中医知识检索与问答系统初始化完成！")

    def run_interactive_cli(self):
        print("=" * 80)
        print(" 中医智能问答系统 (输入'退出'结束对话)")
        print(" 输入 '可视化:您的问题' 或 'viz:您的问题' 来尝试可视化相关子图")
        print(" 输入 'debug:您的问题' 来查看详细的GraphRAG上下文（仅调试用）")
        print("=" * 80)
        while True:
            raw_query = input("\n请输入您的问题: ").strip()
            if not raw_query: continue

            if raw_query.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            include_debug_context = False
            query_to_process = raw_query
            action = "query" # Default action

            if raw_query.lower().startswith("可视化:") or raw_query.lower().startswith("viz:"):
                prefix_len = len("可视化:") if raw_query.lower().startswith("可视化:") else len("viz:")
                query_to_visualize = raw_query[prefix_len:].strip()
                if query_to_visualize:
                    action = "visualize"
                    query_to_process = query_to_visualize
                else:
                    print("请输入要可视化的查询内容。例如：可视化:头痛怎么办")
                    continue
            elif raw_query.lower().startswith("debug:"):
                include_debug_context = True
                query_to_process = raw_query[len("debug:"):].strip()
                print("--- 调试模式开启 (将显示GraphRAG的内部上下文) ---")

            if not query_to_process: # If after stripping prefix, query is empty
                print("请输入有效的问题。")
                continue

            print("\n正在思考中，请稍候...")
            start_time = time.time()
            
            if action == "visualize":
                print(f"正在为查询 '{query_to_process}' 生成知识图可视化...")
                self.visualize_knowledge_for_query(query_to_process)
                # After visualization, allow user to ask about the visualized topic or new topic
                print("--- 可视化完成 ---") 
                continue # Go to next input prompt
            
            # Default action is 'query'
            final_answer = self.query(query_to_process, include_context_debug=include_debug_context)
            
            end_time = time.time()

            print(f"\n💡 智能助手 (用时 {end_time - start_time:.2f}秒):")
            print("-" * 70)
            print(final_answer)
            print("-" * 70)

    def query(self, text: str, include_context_debug: bool = False) -> str:
        # 1. Get GraphRAG response (based on KG)
        # The include_context_debug flag is now passed to generate_graphrag_response_only
        # so it will print its debug info if the flag is true
        graphrag_response, kg_context_str, kg_items, extracted_info = self.rag.generate_graphrag_response_only(text, include_context_debug=include_context_debug)
        
        if include_context_debug: # Additional debug for the app level if needed
            print(f"\n--- [App Debug] GraphRAG原始回答 (传给综合模块前) ---\n{graphrag_response}")
            # print(f"\n--- [App Debug] GraphRAG原始上下文 (传给综合模块前) ---\n{kg_context_str}")
            # print(f"\n--- [App Debug] GraphRAG原始知识项 (传给综合模块前) ---\n{json.dumps(kg_items, ensure_ascii=False, indent=2)}")


        # 2. Get general LLM response (without KG)
        # print("\n正在获取通用模型回答...") # Inform user
        general_response = self.rag.get_general_kimi_response(text)
        # if include_context_debug:
        #     print(f"\n--- [App Debug] 通用大模型回答 (传给综合模块前) ---\n{general_response}")

        # 3. Synthesize
        # print("\n正在综合生成最终回答...") # Inform user
        final_response = self.rag.synthesize_responses(text, graphrag_response, general_response)
        
        return final_response
    
    def visualize_knowledge_for_query(self, query: str):
        extracted_info = self.rag.extract_keywords_and_intent(query)
        keywords = extracted_info.get("keywords", [])
        
        entities_to_visualize = []
        if keywords:
            for keyword in keywords:
                matched = self.kg.search_by_keyword(keyword)
                entities_to_visualize.extend(matched)
        
        unique_entities = list(set(entities_to_visualize))
        
        if not unique_entities:
            print(f"未能从查询 '{query}' 中找到足够精确的实体进行聚焦可视化。尝试显示图谱的随机样本。")
            self.kg.visualize_graph() # Show random sample or whole graph if small
            return

        print(f"将可视化与实体 {unique_entities} 相关的子图...")
        self.kg.visualize_graph(unique_entities)


# # Dummy TOTAL_MAX_BASIC_TRIPLES and MAX_PATHS_TO_SHOW if not defined globally, used in classes.
# # These are now defined inside the methods where they are used.
# TOTAL_MAX_BASIC_TRIPLES = 3 # Example global, but better defined in class or method
# MAX_PATHS_TO_SHOW = 3      # Example global

def main():
    csv_path_from_env = os.getenv("TCM_KG_CSV_PATH", "tcm_kg_data_auto.csv") # Default to a name

    if not os.path.exists(csv_path_from_env):
        print(f"错误: 未找到知识图谱CSV文件 '{csv_path_from_env}'。")
        print("请确保您已运行之前的脚本生成了该文件，或者通过 TCM_KG_CSV_PATH 环境变量指定了正确路径。")
        
        # Provide more specific column names based on typical output of previous script
        sample_columns = ['Subject', 'Predicate', 'Object', 'SourceBookName', 'SourceChapterName']
        print(f"如果需要测试，您可以手动创建一个包含 {', '.join(sample_columns)} 列的简单CSV文件。")

        use_sample_data = input(f"是否创建一个演示用的示例CSV文件 '{csv_path_from_env}' (yes/no)? ").strip().lower()
        if use_sample_data == 'yes':
            print(f"正在创建一个示例CSV文件 '{csv_path_from_env}' 用于演示...")
            sample_data = {
                'Subject': [
                    '治疗方案_逍遥散', '治疗方案_逍遥散', '治疗方案_逍遥散', '柴胡', '当归', '白芍', '乳腺增生', '肝气郁结',
                    '治疗方案_六味地黄丸', '治疗方案_六味地黄丸', '治疗方案_六味地黄丸', '熟地黄','治疗方案_六味地黄丸',
                    '月经不调', '月经不调', '头痛', '生化汤', '生化汤', '失眠', '失眠', '酸枣仁汤', '酸枣仁'
                ],
                'Predicate': [
                    '使用药材', '使用药材', '使用药材', '剂量', '剂量', '剂量', '常见症状', '表现症状',
                    '治疗疾病', '使用药材', '备注', '剂量','属于范畴',
                    '可能病因', '推荐方剂', '定义治疗方案', '使用药材', '治疗疾病', '表现症状', '治疗症状', '使用药材', '剂量'
                ],
                'Object': [
                    '柴胡', '当归', '白芍', '9克', '9克', '12克', '乳房胀痛', '乳腺增生',
                    '肝肾阴虚', '熟地黄', '蜜丸，一次9克，一日2次', '15克','补益剂',
                    '肝气郁结', '治疗方案_逍遥散', '治疗方案_头痛经验方', '当归', '产后血瘀', '心神不宁', '酸枣仁汤', '酸枣仁', '15克'
                ],
                'SourceBookName': ['测试医书'] * 22, # Adjusted count
                'SourceChapterName': ['测试章节'] * 22 # Adjusted count
            }
            sample_df = pd.DataFrame(sample_data)
            try:
                # Ensure directory exists if csv_path_from_env includes a path
                output_dir = os.path.dirname(csv_path_from_env)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                sample_df.to_csv(csv_path_from_env, index=False, encoding='utf-8-sig')
                print(f"示例CSV文件 '{csv_path_from_env}' 已创建。请用您的实际数据替换它以获得最佳效果。")
            except Exception as e_csv:
                print(f"创建示例文件失败: {e_csv}")
                return
        else:
            print("程序将退出，因为缺少必要的输入文件。")
            return

    app = TCMGraphRAGApp(csv_path_from_env)
    app.run_interactive_cli()

if __name__ == "__main__":
    main()
