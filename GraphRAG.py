import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
# from sklearn.metrics.pairwise import cosine_similarity
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

class TCMKnowledgeGraph:
    def __init__(self, csv_path: str = None):
        self.graph = nx.MultiDiGraph()
        self.triples_with_source = [] 
        self.relation_types = set()
        self.entity_types = {}
        self.treatment_plans_details = {}

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
                   source_book: str = "未知来源", source_chapter: str = "未知来源",
                   source_info_str: str = "未知来源"):
        self.graph.add_node(subject)
        self.graph.add_node(obj)
        
        self.graph.add_edge(subject, obj, relation=predicate, source=source_info_str)
        self.relation_types.add(predicate)
        self.triples_with_source.append((subject, predicate, obj, source_book, source_chapter, source_info_str))

        
        treatment_related_predicates = ["使用草药", "剂量", "制备方法", "备注", "具有症状", "描述治疗方案", "治疗疾病", "治疗症状"]
        if predicate in treatment_related_predicates and "治疗方案" in subject:
            if subject not in self.treatment_plans_details:
                self.treatment_plans_details[subject] = {}
            if predicate not in self.treatment_plans_details[subject]:
                self.treatment_plans_details[subject][predicate] = []
           
            self.treatment_plans_details[subject][predicate].append({"value": obj, "source": source_info_str})

   
    def get_entity_relations(self, entity: str, specific_relations: Optional[List[str]] = None) -> List[Tuple[str, str, str, str]]:
        relations = []
        
        for s, p, o, book, chap, source_str in self.triples_with_source:
            if s == entity or o == entity:
                if specific_relations is None or p in specific_relations:
                    relations.append((s, p, o, source_str))
        return relations

   
    def get_treatment_plan_full_details(self, plan_name: str) -> Dict[str, Any]:
        details = {
            "名称": plan_name, "组成": [], "制备方法": [], "功能主治": [],
            "备注": [], "相关症状": [], "治疗疾病": [], "来源信息": set() 
        }

        # 1. Retrieve from preprocessed dictionary and collect sources
        if plan_name in self.treatment_plans_details:
            for pred, obj_source_list in self.treatment_plans_details[plan_name].items():
                for item in obj_source_list:
                    obj_val = item["value"]
                    source_str = item["source"]
                    if source_str != "未知来源": details["来源信息"].add(source_str)

                    if pred == "使用草药":
                        details["组成"].append({"药材": obj_val, "剂量": "未知", "来源": source_str})
                    elif pred == "制备方法": details["制备方法"].append({"value": obj_val, "source": source_str})
                    elif pred == "备注": details["备注"].append({"value": obj_val, "source": source_str})
                    elif pred == "具有症状": details["相关症状"].append({"value": obj_val, "source": source_str})
                    elif pred == "治疗疾病": details["功能主治"].append({"value": obj_val, "source": source_str})


        # 2. Obtain more comprehensive information through graph traversal and extract sources from edge attributes
        for u, target, edge_data in self.graph.out_edges(plan_name, data=True):
            if u != plan_name: continue 
            
            relation = edge_data['relation']
            source_str = edge_data.get('source', "未知来源")
            if source_str != "未知来源": details["来源信息"].add(source_str)

            if relation == "使用草药":
                if not any(d["药材"] == target for d in details["组成"]):
                    details["组成"].append({"药材": target, "剂量": "未知", "来源": source_str})
            elif relation == "制备方法" and not any(d["value"] == target for d in details["制备方法"]):
                details["制备方法"].append({"value": target, "source": source_str})
            elif relation == "备注" and not any(d["value"] == target for d in details["备注"]):
                details["备注"].append({"value": target, "source": source_str})
            elif relation == "治疗疾病" and not any(d["value"] == target for d in details["功能主治"]):
                details["功能主治"].append({"value": target, "source": source_str})
            elif relation == "治疗症状" and not any(d["value"] == target for d in details["相关症状"]):
                details["相关症状"].append({"value": target, "source": source_str})

        # For each medicinal herb, try to search for its "dosage" information and its source
        updated_composition = []
        for herb_item in details["组成"]:
            herb_name = herb_item["药材"]
            herb_dosages_with_source = []
            # Find the source of the triplet (herb name, dosage, dosage-value)
            for s_h, p_h, o_h, book_h, chap_h, source_str_h in self.triples_with_source:
                if s_h == herb_name and p_h == "剂量":
                    herb_dosages_with_source.append({"value": o_h, "source": source_str_h})
                    if source_str_h != "未知来源": details["来源信息"].add(source_str_h)
            
            if herb_dosages_with_source:
                # Simplification: If a medicinal herb has multiple dose records, merge them, and also merge the sources or take the most common ones
                herb_item["剂量"] = "; ".join([ds["value"] for ds in herb_dosages_with_source])
                # Simply add these dosage sources to the sources of medicinal herbs, or use a unified list
                herb_item["剂量来源"] = list(set(ds["source"] for ds in herb_dosages_with_source if ds["source"] != "未知来源"))

            updated_composition.append(herb_item)
        details["组成"] = updated_composition
        
        details["来源信息"] = sorted(list(details["来源信息"])) 
        details = {k: v for k, v in details.items() if v} 
        return details

    def search_by_keyword(self, keyword: str) -> List[str]:
        matched_entities = []
        for entity in self.graph.nodes():
            if keyword.lower() == str(entity).lower() or keyword.lower() in str(entity).lower():
                matched_entities.append(str(entity))
        return list(set(matched_entities))


    def get_related_entities(self, entity: str, relation_type: Optional[str] = None, max_depth: int = 1) -> List[str]:
        if entity not in self.graph.nodes():
            return []
        visited = set([entity])
        queue = [(entity, 0)]
        related_entities = []
        while queue:
            current_entity, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            # For outgoing edges
            for _, neighbor, edge_data in self.graph.out_edges(current_entity, data=True):
                if relation_type and edge_data.get('relation') != relation_type:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    related_entities.append(neighbor)
            # For incoming edges (optional, depending on definition of "related")
            for source, _, edge_data in self.graph.in_edges(current_entity, data=True):
                if relation_type and edge_data.get('relation') != relation_type:
                    continue
                if source not in visited: # source is the neighbor here
                    visited.add(source)
                    queue.append((source, depth + 1))
                    related_entities.append(source)

        return list(set(related_entities))


    def visualize_graph(self, entities: List[str] = None, figsize: Tuple[int, int] = (15, 12)):
        if entities:
            expanded_entities = set(entities)
            for entity in entities:
                if entity in self.graph:
                    for neighbor in nx.neighbors(self.graph, entity): 
                        expanded_entities.add(neighbor)
            g = self.graph.subgraph(list(expanded_entities)).copy()
            # Remove isolated nodes to make visualization more focused on relationships
            g.remove_nodes_from(list(nx.isolates(g)))
        else:
            if len(self.graph.nodes()) > 50:
                nodes = list(self.graph.nodes())
                sampled_nodes = np.random.choice(nodes, size=50, replace=False)
                g = self.get_subgraph(sampled_nodes)
                g.remove_nodes_from(list(nx.isolates(g)))
            else:
                g = self.graph.copy()

        if not g.nodes():
            print("子图为空，无法可视化。")
            return

        plt.figure(figsize=figsize)
        try:
            pos = nx.kamada_kawai_layout(g)
        except Exception: 
             pos = nx.spring_layout(g, k=0.15, iterations=20) 

        nx.draw_networkx_nodes(g, pos, node_size=300, alpha=0.7, node_color='skyblue')
        nx.draw_networkx_edges(g, pos, width=0.8, alpha=0.3, edge_color='gray', arrows=True, arrowstyle='->', arrowsize=10)
        nx.draw_networkx_labels(g, pos, font_size=9, font_family='SimHei')

        edge_labels_dict = {}
        for u, v, data in g.edges(data=True):
            if (u,v) not in edge_labels_dict: 
                 edge_labels_dict[(u,v)] = data.get('relation', '')

        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels_dict, font_size=7, font_family='SimHei', alpha=0.8)

        plt.title("中医知识子图可视化", fontproperties={'family':'SimHei', 'size':16})
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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MOONSHOT_API_KEY}"
        }
        data = {
            "model": "moonshot-v1-32k", 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(MOONSHOT_API_URL, headers=headers, json=data, timeout=120) 
            response.raise_for_status()
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            self.llm_cache[cache_key] = response_text
            return response_text
        except requests.exceptions.Timeout:
            print("调用API超时。")
            return "调用API超时，请稍后再试或检查网络连接。"
        except Exception as e:
            print(f"调用API失败: {e}")
            return f"调用API时发生错误: {str(e)}"

    
    def extract_keywords_and_intent(self, query: str) -> Dict[str, Any]:
        """
        从查询中提取关键词和用户意图（例如，是查找信息还是寻求治疗方案）。
        """
        prompt = f"""
        请分析以下中医相关查询，提取关键实体词（如中药名、症状、疾病名、方剂名等），并判断用户的主要意图。
        主要意图可以是："查询实体信息"、"寻求治疗方案"、"比较实体"、"未知"。
        请按JSON格式输出，包含 "keywords" (字符串列表) 和 "intent" (字符串) 两个字段。

        查询: "{query}"

        JSON输出:
        """
        response_text = self.query_moonshot_api(prompt, temperature=0.1, max_tokens=512)
        try:
            clean_response = re.sub(r"```json\n?|\n?```", "", response_text).strip()
            result = json.loads(clean_response)
            if not isinstance(result.get("keywords"), list): 
                result["keywords"] = []
            if not isinstance(result.get("intent"), str):
                 result["intent"] = "未知"
            return result
        except json.JSONDecodeError:
            print(f"关键词和意图提取JSON解析失败: {response_text}")
            keywords = [kw.strip() for kw in response_text.splitlines() if kw.strip() and not kw.startswith("{")] 
            return {"keywords": keywords if keywords else [query], "intent": "未知"}

    def retrieve_relevant_knowledge(self, query: str, extracted_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        keywords = extracted_info.get("keywords", [])
        intent = extracted_info.get("intent", "未知")
        # print(f"提取的关键词: {keywords}, 用户意图: {intent}")

        relevant_knowledge_items = []
        processed_entities_for_detail = set() 

        all_matched_entities = []
        for keyword in keywords:
            matched = self.kg.search_by_keyword(keyword)
            all_matched_entities.extend(matched)
        unique_entities = list(set(all_matched_entities))
        # print(f"知识图谱中匹配到的实体: {unique_entities}")

        candidate_plans_for_details = []

        # 1. If the intention is to "seek treatment options" or if the query contains treatment-related vocabulary
        is_seeking_treatment_intent = (intent == "寻求治疗方案") or \
                                    any(treat_kw in query.lower() for treat_kw in ["怎么办", "如何治", "用什么药", "治疗方法", "方剂"])

        if is_seeking_treatment_intent:
            # a. Directly matching treatment plans by name
            for entity_name in unique_entities:
                if "治疗方案" in entity_name and entity_name not in processed_entities_for_detail:
                    candidate_plans_for_details.append(entity_name)
                    processed_entities_for_detail.add(entity_name)
            
            # b. Reverse search or forward search for treatment plans through keywords (symptoms/diseases)
            symptom_disease_keywords = [kw for kw in keywords if "治疗方案" not in kw] 
            
            for keyword_sd in symptom_disease_keywords:
                for s, p, o, book, chap, source_str in self.kg.triples_with_source:
                    if o.lower() == keyword_sd.lower() and \
                    ("治疗疾病" in p or "治疗症状" in p or "具有症状" in p):
                        if s not in processed_entities_for_detail:
                            candidate_plans_for_details.append(s)
                            processed_entities_for_detail.add(s)
                    elif s.lower() == keyword_sd.lower() and \
                        ("推荐方剂" in p or "定义治疗方案" in p):
                        if o not in processed_entities_for_detail:
                            candidate_plans_for_details.append(o)
                            processed_entities_for_detail.add(o)
            
            candidate_plans_for_details = list(set(candidate_plans_for_details))
            # print(f"初步候选治疗方案列表 (意图导向): {candidate_plans_for_details}")


        # 2. Obtain detailed information on the identified treatment plan
        MAX_PLANS_TO_DETAIL = 5
        detailed_plans_count = 0
        for plan_name in candidate_plans_for_details:
            if detailed_plans_count >= MAX_PLANS_TO_DETAIL: break
            # print(f"正在为治疗方案 '{plan_name}' 收集详细信息...")
            plan_details = self.kg.get_treatment_plan_full_details(plan_name)
            if plan_details.get("组成") or plan_details.get("功能主治"):
                relevant_knowledge_items.append({"type": "治疗方案", "name": plan_name, "details": plan_details})
                detailed_plans_count += 1
                processed_entities_for_detail.add(plan_name) 

        # 3. Obtain basic triplets related to unique_dentities (schemes for which details have not been obtained) as supplementary information
        if not relevant_knowledge_items or len(relevant_knowledge_items) < MAX_PLANS_TO_DETAIL + 1 or not is_seeking_treatment_intent:
            basic_triples_context_with_source = []
            MAX_TRIPLES_PER_ENTITY = 2
            TOTAL_MAX_BASIC_TRIPLES = 5 
            triples_collected_count = 0

            for entity in unique_entities:
                if entity in processed_entities_for_detail:
                    continue
                if triples_collected_count >= TOTAL_MAX_BASIC_TRIPLES: break

                relations = self.kg.get_entity_relations(entity)
                sorted_relations = self._rank_triples_by_relevance(relations, query, keywords)
                
                added_count_for_entity = 0
                for triple_with_source in sorted_relations:
                    if triples_collected_count >= TOTAL_MAX_BASIC_TRIPLES or added_count_for_entity >= MAX_TRIPLES_PER_ENTITY:
                        break
                    basic_triples_context_with_source.append(triple_with_source)
                    triples_collected_count +=1
                    added_count_for_entity +=1
            
            if basic_triples_context_with_source:
                has_triple_list = any(item.get("type") == "三元组列表" for item in relevant_knowledge_items)
                if not has_triple_list:
                    relevant_knowledge_items.append({"type": "三元组列表", 
                                                    "triples": basic_triples_context_with_source})

        MAX_KNOWLEDGE_ITEMS = 3
        return relevant_knowledge_items[:MAX_KNOWLEDGE_ITEMS]


    def _rank_triples_by_relevance(self, triples_with_source: List[Tuple[str, str, str, str]], 
                                   query: str, keywords: List[str]) -> List[Tuple[str, str, str, str]]:
            scored_triples = []
            for triple_item in triples_with_source:
                s, p, o, _ = triple_item
                score = 0
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower in s.lower(): score += 5
                    if kw_lower in o.lower(): score += 3
                    if kw_lower in p.lower(): score += 1
                for word in query.split():
                    word_lower = word.lower()
                    if word_lower not in [k.lower() for k in keywords]:
                        if word_lower in s.lower(): score += 2
                        if word_lower in o.lower(): score += 1
                scored_triples.append((triple_item, score))
            return [t[0] for t in sorted(scored_triples, key=lambda x: x[1], reverse=True)]

    def _format_knowledge_for_llm(self, knowledge_items: List[Dict[str, Any]]) -> str:
        formatted_text = ""
        if not knowledge_items:
            return "未从知识库中检索到与查询直接相关的详细中医知识。" 

        plan_count = 0
        for item in knowledge_items:
            if item["type"] == "治疗方案":
                plan_count += 1
                details = item["details"]
                formatted_text += f"【治疗方案 {plan_count}: {details.get('名称', item['name'])}】\n" 

                plan_sources = details.get("来源信息", [])
                if plan_sources:
                    source_display = plan_sources[0] if len(plan_sources) == 1 else '; '.join(plan_sources)
                    formatted_text += f"  主要来源: {source_display}\n"

                if details.get("功能主治"):
                    formatted_text += f"  功能主治:\n"
                    for val_src_item in details["功能主治"]:
                        source_tag = f"(来源: {val_src_item['source']})" if val_src_item['source'] != "未知来源" else ""
                        formatted_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                if details.get("相关症状"):
                    formatted_text += f"  相关症状:\n"
                    for val_src_item in details["相关症状"]:
                        source_tag = f"(来源: {val_src_item['source']})" if val_src_item['source'] != "未知来源" else ""
                        formatted_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                if details.get("组成"):
                    formatted_text += "  组成:\n"
                    for comp in details["组成"]:
                        dose_str = comp.get('剂量', '未知')
                        dose_source_str = ""
                        if comp.get("剂量来源"):
                            unique_dose_sources = sorted(list(set(comp['剂量来源']))) 
                            if unique_dose_sources and unique_dose_sources != ["未知来源"]:
                                dose_source_str = f" (剂量来源: {'; '.join(unique_dose_sources)})"
                        
                        herb_source_str = f"(药材条目来源: {comp['来源']})" if comp['来源'] != "未知来源" else ""
                        formatted_text += f"    - {comp['药材']} (剂量: {dose_str}{dose_source_str}) {herb_source_str}\n".strip() + "\n"
                
                for detail_key, display_name in [("制备方法", "制备方法"), ("备注", "备注")]:
                    if details.get(detail_key):
                        formatted_text += f"  {display_name}:\n"
                        for val_src_item in details[detail_key]:
                            source_tag = f"(来源: {val_src_item['source']})" if val_src_item['source'] != "未知来源" else ""
                            formatted_text += f"    - {val_src_item['value']} {source_tag}\n".strip() + "\n"
                formatted_text += "\n"

            elif item["type"] == "三元组列表" and item.get("triples"):
                formatted_text += "【其他相关知识点】:\n"
                for s, p, o, source_str in item["triples"]:
                    source_tag = f"(来源: {source_str})" if source_str != "未知来源" else ""
                    formatted_text += f"  - “{s}” {p} “{o}” {source_tag}\n".strip() + "\n"
                formatted_text += "\n"
        
        if not formatted_text.strip(): 
            return "未从知识库中检索到足够详细的中医知识来回答该问题。"
        return formatted_text.strip()

    def generate_graphrag_response_only(self, query: str, include_context_debug: bool = False) -> Tuple[str, str, List[Dict[str, Any]], Dict[str,Any]]:
        extracted_info = self.extract_keywords_and_intent(query)
        relevant_knowledge_items = self.retrieve_relevant_knowledge(query, extracted_info)
        context_for_llm = self._format_knowledge_for_llm(relevant_knowledge_items) 
        intent = extracted_info.get("intent", "未知")

        no_kg_context_messages = [
            "未从知识库中检索到与查询直接相关的详细中医知识。",
            "未从知识库中检索到足够详细的中医知识来回答该问题。"
        ]
        if context_for_llm.strip() in no_kg_context_messages or not relevant_knowledge_items:
            response_text = context_for_llm 
            # print("GraphRAG: No specific context found in KG.") # 调试信息
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
        2.  如果上下文中包含治疗方案的详细组成、剂量等，请在回答中【明确列出这些细节】。
        3.  在你的回答中，如果引用了“中医知识库上下文”中的具体信息点，请在该信息点或句末【必须】使用括号注明其来源，格式为“(来源:《书籍名》- 章节名)”。
        4.  回答应尽可能【精炼且直接】。如果上下文信息足以回答，请直接给出答案；如果不足，请指出信息有限。
        5.  注意排版，可以使用项目符号。
        """
        is_seeking_treatment_intent = (intent == "寻求治疗方案") or \
                                any(treat_kw in query.lower() for treat_kw in ["怎么办", "如何治", "用什么药", "治疗方法", "方剂"])
        if is_seeking_treatment_intent:
            prompt_template += """
           [关于治疗方案的额外指示]：如果上下文中提到了与用户问题相关的治疗方案，请详细说明其【名称、主要功能主治、组成药材（及剂量，若有）、制备方法/备注（若有）】，并务必标注各项信息的来源。
           """
        prompt_template += "\n请给出你的回答："

        # print(f"--- Prompt for GraphRAG Response Only (Temperature: 0.1) ---") # 调试
        # print(prompt_template) # 调试
        # print(f"-------------------------------------------------------------") # 调试

        response_text = self.query_moonshot_api(prompt_template, temperature=0.1, max_tokens=2000)

        if include_context_debug:
            debug_output = (
                f"---DEBUG: GraphRAG Context Generation---\n"
                f"意图：{intent}\n关键词：{extracted_info.get('keywords')}\n"
                f"检索到的知识项数量：{len(relevant_knowledge_items)}\n"
                f"---CONTEXT FOR GraphRAG-Kimi---\n{context_for_llm}\n"
                f"---GraphRAG-Kimi RESPONSE---\n{response_text}"
            )
            # print(debug_output) 
        
        return response_text, context_for_llm, relevant_knowledge_items, extracted_info        

    def get_general_kimi_response(self, query: str, temperature: float = 0.5, max_tokens: int = 2048) -> str:
        prompt = f"""
        你是一位知识渊博且资深的中医专家。请针对以下用户提出的问题，提供一个全面、详细、且结构清晰的解答。
        请跟据具体问题（药物/病症查询、功能/作用等）尽可能从不同方面（例如：定义、病因病机、主要类型、常见症状、诊断要点、治疗原则、常用方药举例、预后转归、生活调理及注意事项等，根据问题类型酌情选择）且有重点地进行阐述。
        请确保语言专业、严谨，同时也要易于理解。
        
        用户问题： “{query}”

        你的专业解答：
        """
        # print("--- Prompt for General Kimi Response ---") # 调试用
        # print(prompt)
        # print("--------------------------------------")
        return self.query_moonshot_api(prompt, temperature=temperature, max_tokens=max_tokens)
    
    def synthesize_responses(self, query: str, graphrag_response: str, general_response: str,
                             temperature: float = 0.3, max_tokens: int = 3000) -> str:

        is_graphrag_valid = not any(
            msg in graphrag_response for msg in [
                "未从知识库中检索到与查询直接相关的详细中医知识。",
                "未从知识库中检索到足够详细的中医知识来回答该问题。"
            ]
        ) and graphrag_response.strip()

        synthesis_prompt = f"""
        你是一位资深的中医内容编辑和撰稿专家。你的任务是将以下两份关于用户问题“{query}”的回答，巧妙地融合成一个单一、全面、准确、行文流畅自然、且带有清晰来源标注的最终专业答案。

        【第一份回答】（此回答主要基于一个专门的古籍中医知识图谱检索，并由AI初步总结，其核心价值在于包含具体的古籍文献来源信息和细节）：
        ---
        {graphrag_response if is_graphrag_valid else "（知识图谱中未找到与问题直接相关的具体记载）"}
        ---

        【第二份回答】（此回答来自一个通用的大型语言模型，特点是知识面较广，解释可能更系统和详尽，但通常不包含具体的古籍文献来源）：
        ---
        {general_response}
        ---

        请遵循以下【严格的整合指示】进行操作：

        1.  **结构与内容主干**：以【第二份回答】（通用回答）的结构、广度和系统性论述作为整合后答案的基础框架和主要叙述流程。
        2.  **细节融入与深度增强**：
            * 将【第一份回答】（图谱版回答）中的【所有具体的、有价值的信息点，如药物组成、剂量、特定的治疗方法细节、古籍中的独特论述，以及最重要的——文献来源标注】（例如“(来源: 《金匮要略》 - 某某篇)”）——【准确无误且极为自然地】融入到【第二份回答】的相应论述段落或知识点中。
            * **核心目标**：让这些来自古籍的信息看起来像是对通用回答中相应论点的**原生补充、具体例证、或深化解释**，而不是生硬的插入或独立的附加信息块。力求使整合后的文本浑然一体。
        3.  **来源信息的处理与呈现**：
            * 如果两份回答中存在信息重叠（例如，对同一功效或方剂的描述），【优先采用并整合来自第一份回答中带有文献来源的表述和细节】，用其来丰富、具体化或替代通用回答中的对应内容。
            * 所有来自第一份回答的文献来源标注都必须保留，并清晰地附在相应的信息点之后。确保来源格式为“(来源: 《书籍名》 - 章节名)”。来源标注应作为信息点阐述完毕后的自然收尾，避免突兀感。
        4.  **处理图谱无特定信息的情况**：如果【第一份回答】明确指出“未找到信息”或内容为空，则最终答案主要依赖【第二份回答】。此时，无需刻意提及知识图谱未找到信息，除非你认为这样的说明对用户有益。
        5.  **内容补充与文脉扩展**：
            * 如果【第一份回答】提供了【第二份回答】中完全没有提及的【与问题相关的具体治疗方案、药物组成、剂量、古籍观点等重要细节】，请务必将这些有价值的内容【无缝地、合乎逻辑地整合】到最终答案的恰当部分。
            * 这可能需要你对【第二份回答】的局部结构进行微调或扩展，以确保新增信息的融入既自然又保持了整体论述的连贯性和流畅性。
        6.  **专业性、可读性与文风统一**：
            * 最终答案应保持中医的专业术语准确性，同时语言表达应流畅自然、结构清晰，易于普通用户理解（可适当使用项目符号、分点阐述等方式优化排版）。
            * **至关重要的是**：确保整合后的全文文风统一、语调一致，避免出现两种回答风格的明显割裂感，让读者感觉这是由一位专家一气呵成撰写的内容。同时，对于读者问题中的关注点（某种药物/病症的查询、功能、作用、定义等）应在突出相应的重点。
        7.  **标准提醒**：在最终答案的末尾，务必加上标准的用药提醒：“请注意，以上信息仅供参考，具体用药和治疗方案请务必咨询专业中医师进行辨证论治，切勿自行用药。”

        请基于以上所有指示，以专业的判断和高超的编辑技巧，输出整合后的【最终专业答案】：
        """
        # print("--- Prompt for Synthesis ---") # 调试用
        # print(synthesis_prompt)
        # print("----------------------------")
        return self.query_moonshot_api(synthesis_prompt, temperature=temperature, max_tokens=max_tokens)

class TCMGraphRAGApp:
    def __init__(self, csv_path: str):
        if not MOONSHOT_API_KEY:
            print("警告: 未设置MOONSHOT_API_KEY环境变量，API调用将失败")
        print("正在加载中医知识图谱...")
        self.kg = TCMKnowledgeGraph(csv_path)
        self.rag = GraphRAG(self.kg)
        print("中医知识检索与问答系统初始化完成！")

    def run_interactive_cli(self):
        print("=" * 80)
        print(" 中医智能问答系统 (输入'退出'结束对话, 输入 '可视化:您的问题' 来尝试可视化)")
        print(" 输入 'debug:您的问题' 来查看详细的GraphRAG上下文（仅调试用）")
        print("=" * 80)
        while True:
            raw_query = input("\n请输入您的问题: ").strip()
            if not raw_query: continue

            if raw_query.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            include_debug = False
            query_to_process = raw_query

            if raw_query.lower().startswith("可视化:"):
                query_to_visualize = raw_query[len("可视化:"):].strip()
                if query_to_visualize:
                    print(f"正在为查询 '{query_to_visualize}' 生成知识图可视化...")
                    self.visualize_knowledge_for_query(query_to_visualize)
                else:
                    print("请输入要可视化的查询内容。")
                continue
            elif raw_query.lower().startswith("debug:"):
                include_debug = True
                query_to_process = raw_query[len("debug:"):].strip()
                print("--- 调试模式开启 ---")


            print("\n正在思考中，请稍候...")
            start_time = time.time()
            
            final_answer = self.query(query_to_process, include_context_debug=include_debug) 
            
            end_time = time.time()

            print(f"\n💡 智能助手 (用时 {end_time - start_time:.2f}秒):")
            print("-" * 70)
            print(final_answer)
            print("-" * 70)

    def query(self, text: str, include_context_debug: bool = False) -> str: 
        graphrag_response, kg_context_str, kg_items, extracted_info = self.rag.generate_graphrag_response_only(text, include_context_debug=include_context_debug)
        
        if include_context_debug:
             print(f"\n--- 图谱版回答 (GraphRAG Kimi) ---\n{graphrag_response}")
             print(f"\n--- 用于图谱版回答的上下文 ---\n{kg_context_str}")
             # print(f"\n--- 检索到的知识图谱条目 ---\n{json.dumps(kg_items, ensure_ascii=False, indent=2)}")


        general_response = self.rag.get_general_kimi_response(text)
        # if include_context_debug: # 调试时打印
            # print(f"\n--- 通用版回答 (General Kimi) ---\n{general_response}")

        print("\n 生成回答...")
        final_response = self.rag.synthesize_responses(text, graphrag_response, general_response)
        
        return final_response
    
    def visualize_knowledge_for_query(self, query: str):
        extracted_info = self.rag.extract_keywords_and_intent(query)
        keywords = extracted_info.get("keywords", [])
        
        entities_to_visualize = []
        for keyword in keywords:
            matched = self.kg.search_by_keyword(keyword)
            entities_to_visualize.extend(matched)
        
        if not entities_to_visualize:
            print(f"未能从查询 '{query}' 中找到直接匹配的实体进行可视化。")
            # self.kg.visualize_graph()
            return

        unique_entities = list(set(entities_to_visualize))
        print(f"将可视化与实体 {unique_entities} 相关的子图...")
        self.kg.visualize_graph(unique_entities)



def main():
    # set your dataset here
    csv_path_from_previous_script = os.getenv("TCM_KG_CSV_PATH")

    if not os.path.exists(csv_path_from_previous_script):
        print(f"错误: 未找到知识图谱CSV文件 '{csv_path_from_previous_script}'。")
        print("请确保您已运行之前的脚本生成了该文件，或者通过 TCM_KG_CSV_PATH 环境变量指定了正确路径。")
        print("如果需要测试，您可以手动创建一个包含 Subject, Predicate, Object 列的简单CSV文件。")
        use_sample_data = input(f"是否创建一个演示用的示例CSV文件 '{csv_path_from_previous_script}' (yes/no)? ").lower()
        if use_sample_data == 'yes':
            print(f"正在创建一个示例CSV文件 '{csv_path_from_previous_script}' 用于演示...")
            sample_data = {
                'Subject': [
                    '治疗方案_逍遥散', '治疗方案_逍遥散', '治疗方案_逍遥散', '柴胡', '当归', '白芍', '乳腺增生',
                    '治疗方案_六味地黄丸', '治疗方案_六味地黄丸', '治疗方案_六味地黄丸', '熟地黄',
                    '月经不调', '月经不调', '头痛', '生化汤', '生化汤'
                ],
                'Predicate': [
                    '使用草药', '使用草药', '使用草药', '剂量', '剂量', '剂量', '常见症状',
                    '治疗疾病', '使用草药', '备注', '主治',
                    '可能病因', '推荐方剂', '定义治疗方案', '使用草药', '治疗疾病'
                ],
                'Object': [
                    '柴胡', '当归', '白芍', '9克', '9克', '12克', '乳房胀痛',
                    '肝肾阴虚', '熟地黄', '蜜丸，一次9克，一日2次', '滋阴补肾',
                    '肝气郁结', '治疗方案_逍遥散', '治疗方案_头痛方', '当归', '产后血瘀'
                ],

                'SourceBookName': ['测试医书'] * 16,
                'SourceChapterName': ['测试章节'] * 16
            }
            sample_df = pd.DataFrame(sample_data)
            sample_df.to_csv(csv_path_from_previous_script, index=False, encoding='utf-8-sig')
            print(f"示例CSV文件 '{csv_path_from_previous_script}' 已创建。请用您的实际数据替换它以获得最佳效果。")
        else:
            print("程序将退出，因为缺少必要的输入文件。")
            return

    app = TCMGraphRAGApp(csv_path_from_previous_script)
    app.run_interactive_cli()

if __name__ == "__main__":
    main()