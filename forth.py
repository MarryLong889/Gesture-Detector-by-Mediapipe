import cv2
import numpy as np
import mediapipe as mp
import math
from flask import request, Flask, jsonify
import threading
from threading import Lock
import time
import base64
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.start_time = time.time()  # 记录启动时间

shared_variable = {
    "image_result": None
}
lock = Lock()

# 初始化Mediapipe手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  # 静态图像模式
    max_num_hands=1,
    min_detection_confidence=0.75
)

def vector_2d_angle(v1, v2):
    '''求解二维向量的角度'''
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_ = math.degrees(math.acos((v1_x*v2_x + v1_y*v2_y) / 
                                        (((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))))
    except:
        angle_ = 65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_):
    '''获取对应手相关向量的二维角度,根据角度确定手势'''
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle_)
    # ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle_)
    return angle_list

def h_gesture(angle_list):
    '''根据角度列表判断手势'''
    thr_angle = 70.
    thr_angle_thumb = 65.
    thr_angle_s = 55.
    gesture_str = None
    print(angle_list)
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "Rock" # may change to "石头"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "Paper" # may change to "布"
        elif (angle_list[0]>thr_angle_thumb) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "Sciessors" # may change to "剪刀"
        # else:
        #     gesture_str = "Sciessors"
    return gesture_str
    return gesture_str

def process_base64_image(base64_data):
    """处理Base64编码的图片并识别手势"""
    try:
        logger.info("开始处理Base64图片")
        
        # 移除可能的data URI前缀
        # if base64_data.startswith('data:image/'):
        #     base64_data = base64_data.split(',', 1)[1]
        
        # 解码Base64数据
        image_bytes = base64.b64decode(base64_data)
        
        # 转换为numpy数组
        np_array = np.frombuffer(image_bytes, np.uint8)
        
        # 解码图像
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("无法解码图像")
            return "Error: 无法解码图像"
        
        logger.info(f"成功解码图像，尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 颜色空间转换
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用全局初始化的模型处理图像
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 提取手部关键点
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * image.shape[1]
                    y = hand_landmarks.landmark[i].y * image.shape[0]
                    hand_local.append((x, y))
                
                if hand_local:
                    # 计算角度并识别手势
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    logger.info(f"识别出手势: {gesture_str}")
                    return gesture_str
        
        logger.info("未检测到手势")
        return "None"
            
    except Exception as e:
        logger.exception("处理图像时发生异常")
        return f"Error: {str(e)}"

@app.route("/test", methods=["GET"])
def test():
    """简单测试接口，验证服务器正常运行"""
    logger.info("收到/test请求")
    return jsonify({"message": "服务器正常运行", "timestamp": time.time()})

@app.route("/upload_base64", methods=["POST"])
def upload_base64():
    """处理Base64编码的图片上传和识别"""
    try:
        logger.info("收到/upload_base64请求")
        data = request.get_json()
        
        if not data or 'image' not in data:
            logger.warning("请求缺少'image'参数")
            return jsonify({"error": "Missing 'image' parameter in request"}), 400
        
        base64_image = data['image']
        logger.info(f"收到的Base64数据长度: {len(base64_image)}")
        # 移除可能的data URI前缀
        if base64_image.startswith('data:image/'):
            base64_image = base64_image.split(',', 1)[1]        
        # 验证Base64格式
        try:
            base64.b64decode(base64_image, validate=True)
        except Exception as e:
            logger.error(f"Base64格式验证失败: {e}")
            return jsonify({"error": "Invalid base64 format"}), 400
        
        result = process_base64_image(base64_image)
        logger.info(f"识别结果: {result}")
        if result == "Error: 无法解码图像" or result == "None":
            return jsonify({
            "success": False,
            "gesture": result
        })
        with lock:
            shared_variable["image_result"] = result
        
        return jsonify({
            "success": True,
            "gesture": result
        })
        
    except Exception as e:
        logger.exception("处理Base64图片时发生错误")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/status", methods=["GET"])
def status():
    """返回服务器状态信息"""
    try:
        # 构建状态响应
        status_info = {
            "status": "running",
            "timestamp": time.time(),
            "uptime": time.time() - app.start_time if hasattr(app, 'start_time') else 0
        }
        
        return jsonify(status_info)
    except Exception as e:
        logger.error(f"获取状态信息时出错: {e}")
        return jsonify({"error": "Failed to get server status"}), 500

@app.errorhandler(500)
def internal_error(error):
    """处理内部服务器错误"""
    logger.exception("处理请求时发生内部错误")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    """处理未找到的资源"""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """处理不允许的HTTP方法"""
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    try:
        # 导入必要的模块
        from werkzeug.serving import run_simple
        
        # 启动Flask Web服务
        logger.info("启动Flask Web服务，监听端口5000...")
        logger.warning("警告: 这是一个开发服务器，请不要在生产环境中使用。")
        
        # 使用werkzeug的run_simple，启用线程支持
        run_simple(
            hostname='10.10.20.45',
            port=5000,
            application=app,
            threaded=True,
            processes=1,
            use_reloader=False,
            use_debugger=False,
            use_evalex=False
        )
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.exception("程序启动时发生错误")
    finally:
        logger.info("程序已停止")