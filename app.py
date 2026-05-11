from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import mysql.connector
from mysql.connector import pooling
from model_utils import predict_bot_response, train_cartpole_minimal, train_maze_minimal
import os
from dotenv import load_dotenv
import logging
import time
from functools import wraps

# ====== CONFIGURATION ======
load_dotenv()

app = Flask(__name__)

# Configure logging with emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== DATABASE CONFIGURATION ======
if os.getenv('FLASK_ENV') == 'production':
    db_config = {
        'host': os.getenv('MYSQLHOST'),
        'user': os.getenv('MYSQLUSER'),
        'password': os.getenv('MYSQLPASSWORD'),
        'database': os.getenv('MYSQLDATABASE'),
        'port': int(os.getenv('MYSQLPORT', 3306))
    }
    logger.info("🌐 Production mode - using Railway MySQL")
else:
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'ai_project')
    }
    logger.info("💻 Development mode - using local MySQL")

# ====== DATABASE POOL ======
dbpool = None
try:
    dbpool = pooling.MySQLConnectionPool(
        pool_name="ai_pool",
        pool_size=5,
        **db_config
    )
    logger.info("✅ Database pool initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize database pool: {e}")
    dbpool = None

# ====== UTILITY FUNCTIONS ======
def measure_time(f):
    """Декоратор для измерения времени выполнения функции"""
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"⏱️ {f.__name__} executed in {duration:.2f}s")
        return result
    return decorated

def db_query(query, params=(), fetch=False):
    """
    Безопасное выполнение SQL запросов с обработкой ошибок
    
    Args:
        query: SQL запрос
        params: Параметры для параметризованного запроса
        fetch: Возвращать ли результаты
    
    Returns:
        Результаты запроса (если fetch=True) или None
    """
    if not dbpool:
        logger.error("❌ Database pool not initialized")
        raise Exception("Database connection pool not available")
    
    conn = None
    cursor = None
    try:
        conn = dbpool.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        res = cursor.fetchall() if fetch else None
        conn.commit()
        logger.debug(f"✅ Query executed successfully")
        return res
    except mysql.connector.Error as err:
        logger.error(f"❌ Database error: {err}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ====== ROUTES ======

@app.route('/')
@measure_time
def index():
    """Главная страница со статистикой агентов"""
    try:
        stats = db_query("SELECT * FROM agent_stats", fetch=True)
        logger.info(f"🎯 Loaded {len(stats) if stats else 0} agent statistics")
        return render_template('index.html', stats=stats)
    except Exception as e:
        logger.error(f"❌ Error loading statistics: {e}")
        return render_template('index.html', stats=[], error=str(e)), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """JSON API для получения статистики"""
    try:
        stats = db_query("SELECT * FROM agent_stats", fetch=True)
        return jsonify({
            'success': True,
            'data': stats,
            'count': len(stats) if stats else 0
        })
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/train_view', methods=['GET'])
def train_view():
    """
    Поток обучения агента с Server-Sent Events (SSE)
    
    Параметры query:
        type: 'cartpole' или 'maze' (обязательный)
        episodes: количество эпизодов (1-1000, по умолчанию 50)
        reset: 'true' или 'false' - сбросить ли Q-таблицу (по умолчанию false)
    """
    mode = request.args.get("type", "").lower()
    reset_param = request.args.get("reset", "false").lower() == "true"
    
    # Валидация mode
    if mode not in ['cartpole', 'maze']:
        logger.warning(f"⚠️ Invalid mode requested: {mode}")
        return jsonify({"error": "Invalid mode. Use 'cartpole' or 'maze'"}), 400
    
    # Валидация episodes
    try:
        episodes = int(request.args.get("episodes", 50))
        if episodes < 1 or episodes > 1000:
            logger.warning(f"⚠️ Invalid episodes count: {episodes}")
            return jsonify({"error": "Episodes must be between 1 and 1000"}), 400
    except ValueError:
        logger.warning(f"⚠️ Invalid episodes parameter type")
        return jsonify({"error": "Episodes parameter must be an integer"}), 400
    
    logger.info(f"🚀 Starting training: mode={mode}, episodes={episodes}, reset={reset_param}")

    def generate():
        """Генератор для SSE потока"""
        try:
            action_text = "🔄 Переучивание" if reset_param else "📚 Обучение"
            yield f"data: [START] {action_text} {mode.upper()} на {episodes} эпизодов...\n\n"
            logger.info(f"🎮 {action_text} {mode} начато")
            
            # Обучение агента
            if mode == 'cartpole':
                result_reward = train_cartpole_minimal(episodes, reset_qtable=reset_param)
            else:
                result_reward = train_maze_minimal(episodes, reset_qtable=reset_param)
            
            logger.info(f"🏆 Training completed with reward: {result_reward}")
            
            # Обновление статистики в БД
            try:
                db_query("""
                    UPDATE agent_stats 
                    SET total_episodes = total_episodes + %s, 
                        best_reward = GREATEST(best_reward, %s) 
                    WHERE agent_name = %s
                """, (episodes, float(result_reward), mode))
                logger.info(f"📊 Database updated for {mode}")
                
                yield f"data: [SUCCESS] 🎉 Рекорд сессии: {result_reward} pts\n\n"
                yield f"data: [INFO] 📊 Статистика MySQL обновлена\n\n"
            except Exception as db_error:
                logger.error(f"❌ Database update failed: {db_error}")
                yield f"data: [WARNING] ⚠️ Обучение завершено, но ошибка БД: {str(db_error)}\n\n"
        
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
            yield f"data: [ERROR] ❌ Ошибка: {str(e)}\n\n"
        
        finally:
            yield "data: [DONE] ✅ Завершено\n\n"
            logger.info(f"✅ Training stream finished for {mode}")
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint для Railway"""
    try:
        # Проверка подключения к БД
        db_query("SELECT 1")
        logger.debug("🟢 Health check passed")
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        }), 200
    except Exception as e:
        logger.error(f"🔴 Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'database': 'disconnected',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Обработчик ошибки 404"""
    logger.warning(f"⚠️ 404 Not Found: {request.path}")
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработчик ошибки 500"""
    logger.error(f"❌ 500 Internal Server Error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ====== MAIN ======
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV') != 'production'
    
    logger.info(f"🚀 Starting Flask app on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    
    app.run(
        debug=debug_mode,
        port=port,
        threaded=True,
        host='0.0.0.0'
    )
