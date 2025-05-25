import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv


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
    csv_path = os.getenv("TCM_KG_CSV_PATH", r"C:\Users\13625\Desktop\OpenTCM_demo\data\tcm_KG.csv")
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


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/chat_page')
def chat_ui_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat_api():
    global tcm_app_instance
    if not tcm_app_instance:
        logger.error("API 调用失败：tcm_app_instance 未初始化。")
        return jsonify({"error": "中医问答服务核心组件未正确初始化，请检查后端日志。"}), 503 # Service Unavailable

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        logger.warning("API 调用错误：缺少 'query' 参数。")
        return jsonify({"error": "Query is missing"}), 400

    logger.info(f"收到用户查询: {user_query}")
    try:

        final_answer = tcm_app_instance.query(user_query) 
        logger.info(f"为查询 '{user_query}' 生成的回答长度: {len(final_answer)} 字符")
        return jsonify({"response": final_answer})
    except Exception as e:
        logger.error(f"处理查询 '{user_query}' 时发生严重错误: {e}", exc_info=True)
        return jsonify({"error": "处理您的请求时发生内部错误，请稍后再试或联系管理员。"}), 500

@app.route('/static/images/<filename>')
def serve_image(filename):
    return send_from_directory(os.path.join(app.static_folder, 'images'), filename)


if __name__ == '__main__':
    logger.info("启动 Flask 开发服务器...")
    app.run(host='0.0.0.0', port=8000, debug=True) # debug=True 