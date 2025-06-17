import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
import json
import time


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from GraphRAG import TCMGraphRAGApp
    TCM_RAG_APP_LOADED = True
except ImportError as e:
    logger.error(f"无法导入 TCMGraphRAGApp: {e}. 请确保 GraphRAG.py 文件存在且无误。")
    TCM_RAG_APP_LOADED = False
    TCMGraphRAGApp = None 


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app) 


tcm_app_instance = None

if TCM_RAG_APP_LOADED:
    csv_path = os.getenv("TCM_CSV_PATH", r"tcm_KG.csv") 
    api_key = os.getenv("MOONSHOT_API_KEY")

    if not api_key:
        logger.warning("警告: 未设置 MOONSHOT_API_KEY 环境变量，API 调用将失败。")
    
    if not os.path.exists(csv_path):
        logger.error(f"错误: 知识图谱 CSV 文件 '{csv_path}' 未找到!")
    
    if api_key and os.path.exists(csv_path) and TCMGraphRAGApp:
        try:
            logger.info(f"正在初始化 TCMGraphRAGApp，使用 CSV: {csv_path}...")
            tcm_app_instance = TCMGraphRAGApp(csv_path=csv_path)
            logger.info("TCMGraphRAGApp 初始化成功。")
        except Exception as e:
            logger.error(f"TCMGraphRAGApp 初始化失败: {e}", exc_info=True)
            tcm_app_instance = None
    else:
        logger.warning("因缺少 CSV 文件、API 密钥或 TCMGraphRAGApp 未加载，服务核心功能可能受限。")
else:
    logger.error("TCM RAG App 模块未加载，核心聊天功能将不可用。")



def generate_step_by_step_streaming_response(query):
    """
    生成分步流式响应，将"思考过程"暴露给前端。
    """
    def yield_json(data):
        """辅助函数，用于生成 SSE 格式的 JSON 数据"""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    try:
        rag_instance = tcm_app_instance.rag

  
        yield yield_json({'type': 'thinking_start', 'title': '🤔 思考中...'})


        yield yield_json({'type': 'thinking_step', 'step': 1, 'title': '分析问题意图和关键词'})
        extracted_info = rag_instance.extract_keywords_and_intent(query)
        yield yield_json({'type': 'thinking_content', 'step': 1, 'content': f"意图: {extracted_info.get('intent', '未知')}\n关键词: {', '.join(extracted_info.get('keywords', []))}"})


        yield yield_json({'type': 'thinking_step', 'step': 2, 'title': '查询中医知识图谱'})
        relevant_knowledge_items = rag_instance.retrieve_relevant_knowledge(query, extracted_info)
        context_for_llm = rag_instance._format_knowledge_for_llm(relevant_knowledge_items)
        graphrag_stream = rag_instance.generate_graphrag_response_stream(query, context_for_llm, extracted_info.get('intent', '未知'))
        
        graphrag_response_chunks = []
        for chunk in graphrag_stream:
            graphrag_response_chunks.append(chunk)
            yield yield_json({'type': 'thinking_content', 'step': 2, 'content': chunk})
        graphrag_response = "".join(graphrag_response_chunks)

    
        yield yield_json({'type': 'thinking_step', 'step': 3, 'title': '通用中医知识补充'})
        general_stream = rag_instance.get_general_kimi_response_stream(query)
        
        general_response_chunks = []
        for chunk in general_stream:
            general_response_chunks.append(chunk)
            yield yield_json({'type': 'thinking_content', 'step': 3, 'content': chunk})
        general_response = "".join(general_response_chunks)

        
        yield yield_json({'type': 'thinking_end'})

      
        yield yield_json({'type': 'final_answer_start', 'title': '💡 综合分析结果'})
        final_answer_stream = rag_instance.synthesize_responses(query, graphrag_response, general_response)
        
        for chunk in final_answer_stream:
            yield yield_json({'type': 'final_answer_content', 'content': chunk})
        
      
        disclaimer = "\n\n📝 **重要提醒：** 基于GraphRAG得到的知识有精确来源，其他内容仅供参考。注意，OpenTCM不是真正的中医，如需看病请前往医院就诊。"
        yield yield_json({'type': 'final_answer_content', 'content': disclaimer})
        
        
        yield yield_json({'type': 'final_end'})

    except Exception as e:
        logger.error(f"分步流式响应生成错误: {e}", exc_info=True)
        yield yield_json({
            'type': 'error',
            'content': f"处理您的请求时发生内部错误: {str(e)}"
        })


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/chat_page')
def chat_ui_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['GET'])
def handle_chat_api():
    global tcm_app_instance
    if not tcm_app_instance:
        logger.error("API 调用失败：tcm_app_instance 未初始化。")
        
        def error_stream():
            error_data = {'type': 'error', 'content': '服务核心组件未初始化'}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    
    user_query = request.args.get('query')

    if not user_query:
        logger.warning("API 调用错误：缺少 'query' 参数。")
        def error_stream():
            error_data = {'type': 'error', 'content': '查询内容不能为空'}
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        return Response(error_stream(), mimetype='text/event-stream')

    logger.info(f"收到用户查询: {user_query}")
    
    return Response(
        generate_step_by_step_streaming_response(user_query), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
    )
    
@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.static_folder, 'images'), filename)


if __name__ == '__main__':
    logger.info("启动 Flask 开发服务器...")
    app.run(host='0.0.0.0', port=8000, debug=True)
